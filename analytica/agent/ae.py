import analytica.agent.prompts
from lllm.llm import Agent, LLMCaller, LLMResponder, Dialog, Prompts, find_model_card
from analytica.const import Task, Proposition, Proof, Program, JupyterCell, JupyterCellType, Failure
from lllm.llm import PROMPT_REGISTRY as PR
from lllm.const import APITypes
from lllm.log import build_log_base, LogBase
from typing import Dict, Any, List, Tuple, Union, Optional
from enum import Enum
from lllm.sandbox import JupyterSandbox, JupyterSession
from dataclasses import dataclass, field
from lllm.models import Prompt, ParseError
from analytica.proxy.re import Proxy, Query
import datetime as dt
import random
import analytica.utils as U
import uuid
from analytica.agent.dr import query_dr, DeepResearchModels
from lllm.utils import PrintSystem, StreamWrapper
import inspect
import numpy as np
import asyncio
import os
import shutil


U.cprint(f'{len(PR)} prompts loaded', 'g')


class AgentType(Enum): # AE: Agentical Engine
    RANDOM = 'random'
    VANILLA = 'vanilla'
    ASYNC_VANILLA_PROVER = 'async_vanilla_prover'
    PROVER = 'prover'
    ASYNC_PROVER = 'async_prover'   
    ANALYTICA = 'analytica'
    ANALYTICA_LINEAR = 'analytica_linear'
    ANALYTICA_LOGICAL = 'analytica_logical'
    DEEP_RESEARCH_PROVER = 'deep_research_prover'
    ASYNC_DEEP_RESEARCH_PROVER = 'async_deep_research_prover'


class ProverType(Enum):
    VANILLA = 'vanilla'
    DEEP_RESEARCH = 'deep_research'
    PROGRAMMING = 'programming'


@dataclass  
class Decision:
    option_idx: int
    report: str
    weight: float
    cost: float = None
    proposition: Proposition = None

    def __str__(self):
        _str = f'Option {self.option_idx+1} confidence: {self.weight:.2f}'
        if self.cost is not None:
            _str += f' ({self.cost:.2f})'
        return _str+'\n'

    def to_dict(self):
        if isinstance(self.report, str):
            _report = self.report
        else:
            raise ValueError("Not supported report type")
        return {
            'option_idx': self.option_idx,
            'report': _report,
            'weight': self.weight if self.proposition is None else self.proposition.proof.p_true,
            'cost': self.cost,
            'proposition': self.proposition.to_dict() if self.proposition else None
        }
    
    @classmethod
    def from_dict(cls, d: dict, ckpt_dir: str = None):
        proposition = Proposition.from_dict(d['proposition'], ckpt_dir=ckpt_dir) if d['proposition'] else None
        weight = proposition.proof.p_true if proposition is not None else d['weight']
        return cls(
            option_idx=d['option_idx'], 
            report=d['report'], 
            weight=weight, 
            cost=d['cost'], 
            proposition=proposition
        )

    @classmethod
    def from_proposition(cls, option_idx: int, proposition: Proposition, cost: float = None):
        proof = proposition.proof
        if proof is None:
            raise ValueError("Proposition has no proof")
        return cls(option_idx=option_idx, report=proof.proof, weight=proof.p_true, cost=cost, proposition=proposition)
    
    


@dataclass
class Score:
    soft: float # weighted average of the decisions
    hard: float # winner takes all
    binary: float # binary score
    best: float
    worst: float
    mean: float # mean of dataset
    median: float # median of dataset

    @property
    def norm_term(self):
        return self.best - self.worst

    @property
    def soft_normalized(self):
        return 100*(self.soft - self.worst) / self.norm_term
    
    @property
    def hard_normalized(self):
        return 100*(self.hard - self.worst) / self.norm_term
    
    @property
    def accuracy(self):
        return 100*self.binary 
    
    @property
    def mean_normalized(self):
        return 100*(self.mean - self.worst) / self.norm_term
    
    @property
    def median_normalized(self):
        return 100*(self.median - self.worst) / self.norm_term
    
    def to_str(self, include_stats: bool = False):
        _str = f'''
Soft NPV: {self.soft:.2f} (Normalized: {self.soft_normalized:.2f}%)
Hard NPV: {self.hard:.2f} (Normalized: {self.hard_normalized:.2f}%)
Accuracy: {self.accuracy:.2f}%
'''
        if include_stats:
            _str += f'''
Range: {self.worst:.2f} ~ {self.best:.2f}
Mean NPV: {self.mean:.2f} (Normalized: {self.mean_normalized:.2f}%)
Median NPV: {self.median:.2f} (Normalized: {self.median_normalized:.2f}%)
'''
        return _str

    def short_str(self, normalize: bool = True):
       if normalize:
           return f'Accu: {self.accuracy:.2f}% | Soft: {self.soft_normalized:.2f}% | Hard: {self.hard_normalized:.2f}%'
       else:
           return f'Accu: {self.accuracy:.2f}% | Soft: {self.soft:.2f} | Hard: {self.hard:.2f}'

    def __str__(self):
        return self.to_str()




@dataclass
class Report:
    task: Task
    _decisions: List[Decision]
    threshold: Optional[float] = None

    @property
    def weights(self):
        _weight_sum = sum(d.weight for d in self.decisions)
        return [d.weight / _weight_sum for d in self.decisions]

    @property
    def soft_npv(self):
        _weight_sum = sum(d.weight for d in self.decisions)
        return sum(d.weight * self.task.eval(d.option_idx) for d in self.decisions) / _weight_sum
    
    @property
    def max_idx(self):
        if len(self.decisions) == 2 and self.threshold is not None:
            if self.decisions[0].weight > self.threshold:
                return self.decisions[0].option_idx
            return self.decisions[1].option_idx
        top_decision = max(self.decisions, key=lambda d: d.weight)
        return top_decision.option_idx

    @property
    def hard_npv(self):
        return self.task.eval(self.max_idx)

    @property
    def binary_npv(self):
        return 1 if self.max_idx == self.task.answer else 0
    
    @property
    def decisions(self):
        _decisions = self._decisions
        _decision_idx = {d.option_idx: d for d in _decisions}
        for option in self.task.options: # use counter to fill in the missing options
            _counter = self.task.find_counter(option.index)
            need_fix = False
            if option.index not in _decision_idx:
                need_fix = True
            else:
                _decision = _decision_idx[option.index]
                if _decision.proposition is None and (_decision.weight == 0 or _decision.weight == 1) and _decision.report == option.title:
                    need_fix = True
            if not need_fix:
                continue
            counter_decision = _decision_idx[_counter.index]
            assert _counter is not None, f"Decision for option {option.index} not found"
            assert _counter.index in _decision_idx, f"Decision for option {option.index} and {_counter.index} not found" # this should not happen
            if option.index not in _decision_idx:
                _decisions.append(Decision(
                    option_idx=option.index, report=option.title, weight=1-counter_decision.weight))
            else:
                _decision = _decision_idx[option.index]
                _decision.report = option.title
                _decision.weight = 1-counter_decision.weight
        return _decisions

    @property
    def score(self) -> Score:
        assert self.task.evaluable, "Task is not evaluable at this moment"
        return Score(soft=self.soft_npv, hard=self.hard_npv, binary=self.binary_npv, 
                     best = self.task.best_npv, worst = self.task.worst_npv, 
                     mean = self.task.mean_npv, median = self.task.median_npv)

    def __str__(self) -> str:
        _str = f'{self.task.prompt}\n\nDecisions:\n'
        for d in self.decisions:
            _str += f'{d}'
            if self.task.evaluable:
                _str += f'  (NPV: {self.task.npv(d.option_idx):.2f})'
                if d.option_idx == self.task.answer:
                    _str += f' [Optimal]'
            _str += '\n'
        if self.task.evaluable:
            _str += f'\nScore: \n{self.score}\n\n'
            _task_stats = self.task.npv_stats
            _str += f'Task NPV range: {_task_stats["min"]:.2f} ~ {_task_stats["max"]:.2f} (mean: {_task_stats["mean"]:.2f})\n'
        return _str
    
    def save(self, path: str):
        d = {
            'task': self.task.to_dict(),
            'decisions': [d.to_dict() for d in self._decisions]
        }
        U.save_json(path, d)
    
    @classmethod
    def load(cls, path: str, ckpt_dir: str = None):
        d = U.load_json(path)
        return cls(Task.from_dict(d['task']), [Decision.from_dict(d, ckpt_dir=ckpt_dir) for d in d['decisions']])

    @property
    def propositions(self) -> List[Proposition]:
        return [d.proposition for d in self.decisions]


@dataclass
class Result:
    reports: List[Report]

    @property
    def score(self) -> Score:
        return Score(
            soft=np.mean([r.score.soft for r in self.reports]),
            hard=np.mean([r.score.hard for r in self.reports]),
            binary=np.mean([r.score.binary for r in self.reports]),
            best = np.mean([r.score.best for r in self.reports]),
            worst = np.mean([r.score.worst for r in self.reports]),
            mean = np.mean([r.score.mean for r in self.reports]),
            median = np.median([r.score.median for r in self.reports])
        )

    @property
    def brier_score(self):
        brier_scores = []
        for report in self.reports:
            y = np.array([1 if i == report.task.answer else 0 for i in range(len(report.weights))])
            pred = np.array(report.weights)
            brier_scores.append(np.mean((pred - y) ** 2))
        return np.mean(brier_scores), np.std(brier_scores)

    @property
    def cross_entropy(self):
        cross_entropies = []
        for report in self.reports:
            correct_class_index = report.task.answer
            predicted_probabilities = np.asarray(report.weights)
            epsilon = 1e-15
            p = np.clip(predicted_probabilities[correct_class_index], epsilon, 1. - epsilon)
            cross_entropies.append(-np.log(p))
        return np.mean(cross_entropies), np.std(cross_entropies)
    
    @property
    def confidence(self):
        confidences = []
        for report in self.reports:
            confidences.append(np.max(report.weights))
        return np.mean(confidences), np.std(confidences)

    def calibration_plot(self):
        raise NotImplementedError("Subclass must implement this method")

    def __str__(self) -> str:
        return f'({len(self.reports)} benchmarked reports) {self.score}'

    def short_str(self, normalize: bool = True):
        return f'(# {len(self.reports)}) {self.score.short_str(normalize)}'


class AgentBase:
    agent_type: AgentType = None
    agent_group: List[str] = None # it maps to the agent_configs in config for better reuse 
    is_async: bool = False

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream = None): # use a name extension to distinguish different runs
        if stream is None:
            stream = PrintSystem()
        self.config = config
        assert self.agent_group is not None, f"Agent group is not set for {self.agent_type}"
        _agent_configs = config['agent_configs']
        self.agent_configs = {}
        for agent_name in self.agent_group:
            assert agent_name in _agent_configs, f"Agent {agent_name} not found in agent configs"
            self.agent_configs[agent_name] = _agent_configs[agent_name]
        self._stream = stream
        self._stream_backup = stream
        self.st = None # to be initialized when calling __call__
        self.ckpt_dir = ckpt_dir
        self._log_base = build_log_base(config)
        self.agents = {}
        self.llm_caller = LLMCaller(self.config)
        self.llm_responder = LLMResponder(self.config)
        for agent_name, model_config in self.agent_configs.items():
            model_config = model_config.copy()
            model_name = model_config.pop('model_name')
            self.model = find_model_card(model_name)
            system_prompt_path = model_config.pop('system_prompt_path')
            _api_type = APITypes(model_config.pop('api_type', 'completion'))
            if _api_type == APITypes.COMPLETION:
                _caller = self.llm_caller
            elif _api_type == APITypes.RESPONSE:
                _caller = self.llm_responder
            else:
                raise ValueError(f"Unsupported API type: {_api_type}")
            self.agents[agent_name] = Agent(
                name=agent_name,
                system_prompt=PR[system_prompt_path],  # TODO: directly from prompt
                model=model_name,
                llm_caller=_caller,
                model_args=model_config,
                log_base=self._log_base,
                max_exception_retry=self.config.get('max_exception_retry', 3),
                max_interrupt_times=self.config.get('max_interrupt_times', 5),
                max_llm_recall=self.config.get('max_llm_recall', 0),
            )
        assert self.agent_type is not None, "Agent type is not set"
        self.long_only = config['long_only']
        self.one_side_only = config['one_side_only']
        self.threshold = config.get('threshold', None)

        self.__additional_args = {}
        # check if any args in call besides query and **kwargs
        for arg in inspect.signature(self.call).parameters:
            if arg != 'query' and arg != '**kwargs':
                # get the default value, None if not provided
                self.__additional_args[arg] = inspect.signature(self.call).parameters[arg].default

    def set_st(self, session_name: str):
        self.st = StreamWrapper(self._stream, self._log_base, session_name)

    def restore_st(self):
        self.st = None

    def silent(self):
        self._stream = PrintSystem(silent=True)

    def restore(self):
        self._stream = self._stream_backup  

    def call(self, query: Query, **kwargs) -> Report:
        raise NotImplementedError("Subclass must implement this method")
    
    def config_query(self, query: Query):
        query.task.config(long_only=self.long_only, one_side_only=self.one_side_only)

    def __call__(self, query: Query, session_name: str = None, **kwargs) -> Report:
        if session_name is None:
            session_name = query.task.title.replace(' ', '+')+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        # unsafe in multi-threading, so we use a clone in multi-threading
        self.set_st(session_name)
        self.config_query(query)
        report = self.call(query, **kwargs)
        with self.st.expander('Prediction Overview', expanded=True):
            self.st.code(f'{report}')
        self.restore_st()
        return report
    

class AsyncAgentBase(AgentBase):
    is_async: bool = True

    async def call(self, query: Query, **kwargs) -> Report:
        raise NotImplementedError("Subclass must implement this method")

    async def __call__(self, query: Query, session_name: str = None, **kwargs) -> Report:
        if session_name is None:
            session_name = query.task.title.replace(' ', '+')+'_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        # unsafe in multi-threading, so we use a clone in multi-threading
        self.set_st(session_name)
        self.config_query(query)
        report = await self.call(query, **kwargs)
        with self.st.expander('Prediction Overview', expanded=True):
            self.st.code(f'{report}')
        self.restore_st()
        return report   

    
class Random(AgentBase):
    agent_type: AgentType = AgentType.RANDOM
    agent_group: List[str] = []

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream):
        super().__init__(config, ckpt_dir, stream)
    
    def call(self, query: Query, **kwargs) -> Report:
        task = query.task
        decisions = []
        _options = task.get_options()
        with self.st.status(f'Randomly generating weights for all {len(_options)} options', expanded=False):
            for option in _options:
                weight = random.random()
                decisions.append(Decision(option_idx=option.index, report=option.title, weight=weight))
                self.st.write(f'Option {option.index+1}: {option.title} -> Weight: {weight}')
        return Report(task=task, _decisions=decisions, threshold=self.threshold)


class Vanilla(AgentBase):
    agent_type: AgentType = AgentType.VANILLA
    agent_group: List[str] = ['vanilla_prover']

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream, **kwargs):
        super().__init__(config, ckpt_dir, stream)
        self.agent: Agent = self.agents['vanilla_prover']
        self.prompts = Prompts('vanilla')

    def call(self, query: Query, sequential: bool = True, **kwargs) -> Report:
        # system prompt
        task = query.task
        dialog: Dialog = self.agent.init_dialog()
        _dialog = dialog
        self.st.write(f'{task.prompt}')

        with self.st.expander('System Prompt', expanded=False):
            self.st.write(f'{dialog.tail.content}')
        decisions = []

        # task query
        _options = task.get_options()
        # U.cprint(f'Analyzing {len(_options)} options', color='y')
        for idx, option in enumerate(_options):
            if not sequential:
                _dialog = dialog.fork()
            proposition = query.init_analysis(idx)
            with self.st.status(f'Querying the agent for Option {idx+1}: {option.title}', expanded=False):
                _message = self.agent.send_message(
                    _dialog,
                    self.prompts('vanilla_prover_task'),
                    {
                        'proposition': proposition.prompt,
                        'date': proposition.date
                    },
                )
                self.st.write(f'{_message.content}')
            with self.st.status(f'Agent is analyzing Option {idx+1}...', expanded=True):
                _response, _dialog, _ = self.agent.call(_dialog)
                parsed = _response.parsed
                report = parsed['raw']
                p_true = parsed['p_true']
                key_factor = parsed['key_factor']
                self.st.write(f'{report}')
                self.st.markdown(f'**Truth Value: {p_true:.2f}**\n**Key Factor: {key_factor}**')
                _dialog.overview(remove_tail=True, stream=self.st)
            decisions.append(Decision(option_idx=option.index, report=report, weight=p_true))
            dlg_dir = U.pjoin(self.ckpt_dir, 'dialogs', f'{proposition.pid}')
            U.mkdirs(dlg_dir)
            U.save_json(U.pjoin(dlg_dir, f'{idx}.json'), _dialog.to_dict())
        report = Report(task=task, _decisions=decisions, threshold=self.threshold)
        return report



@dataclass
class ProofStep:  # each step is writing a bunch of cells
    initial_cells: List[JupyterCell]
    debug_trace: List[Tuple[int, str]] = field(default_factory=list)
    program: Program = None

    def __post_init__(self):
        if self.program is None:
            self.program = Program(cells=self.initial_cells)

    @property
    def program_updates(self) -> str:
        updates = {}
        for index, code in self.debug_trace:
            updates[index] = code
        _str = ''
        for index, updated_code in updates.items():
            initial_code = self.initial_cells[index].source
            code_short = initial_code[:200]
            if len(code_short) < len(initial_code):
                code_short += '...'
            _str += f'''Cell {index}:

```python
{code_short}
```

has been updated to:

```python
{updated_code}
```
'''
        return _str
    
    @property
    def failure(self) -> Failure:
        return self.program.failure
    
    @property
    def error_info(self) -> str:
        return self.program.error_info
    
    def test(self, jupyter_session: JupyterSession):
        assert self.program is not None, f"Program is not set"
        return self.program.test(jupyter_session)

    def debug(self, code: str, jupyter_session: JupyterSession) -> bool:
        assert jupyter_session is not None, f"Jupyter session is not set"
        failure_index = self.program.failure.index
        self.debug_trace.append((failure_index, code))
        self.program.debug(code, jupyter_session)
        return self.failure is None 
    
    def write(self, jupyter_session: JupyterSession):
        self.program.write(jupyter_session)

    def to_dict(self):
        return {
            'initial_cells': [c.to_dict() for c in self.initial_cells],
            'debug_trace': self.debug_trace,
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        _initial_cells = [JupyterCell.from_dict(c) for c in d['initial_cells']]
        _debug_trace = [(c['index'], c['source']) for c in d['debug_trace']]
        _program = Program(cells=_initial_cells)
        return cls(initial_cells=_initial_cells, debug_trace=_debug_trace, program=_program)
    
    @classmethod
    def from_program(cls, program: Program):
        return cls(initial_cells=program.cells, debug_trace=[], program=program)

    def send(self, dialog: Dialog, creator: str = 'user') -> Dialog:
        return self.program.send(dialog, creator=creator)
    

@dataclass
class ProofState: # used for checkpointing, maintain the state of the proof
    path: str
    proposition: Proposition
    dialog: Dialog
    steps: List[ProofStep] = field(default_factory=list)
    jupyter_session: JupyterSession = None
    proxy: Proxy = None
    terminate_notebook: bool = False

    def __post_init__(self):
        self.backup()

    def backup(self):
        self.dialog_backup = self.dialog.fork()

    def new_step(self,cells: List[Tuple[str, str]], dialog: Dialog, creator: str = 'user'): # tuples of (tag, content)
        self.backup()
        _cells = []
        for tag, content in cells:
            _type = JupyterCellType.CODE if tag == 'python_cell' else JupyterCellType.MARKDOWN
            _cells.append(JupyterCell(index=len(_cells), source=content, type=_type))
        self.steps.append(ProofStep(initial_cells=_cells))
        success = self.latest_step.test(self.jupyter_session)
        dialog = self.send_latest(dialog, creator=creator)
        return success, dialog
    
    def revert(self):
        self.dialog = self.dialog_backup
        self.latest_step.program.clear(self.jupyter_session)
        self.steps.pop()

    def debug(self, code: str, dialog: Dialog, creator: str = 'user') -> bool:
        success = self.latest_step.debug(code, self.jupyter_session)
        dialog = self.send_latest(dialog, creator=creator)
        return success, dialog

    def step_done(self):
        self.save()

    def send_latest(self, dialog: Dialog, creator: str = 'user') -> Dialog:
        return self.latest_step.send(dialog, creator=creator)
    
    @property
    def notebook_dir(self) -> str:
        return self.jupyter_session.dir

    @property
    def latest_program(self) -> Program:
        return self.latest_step.program
    
    @property
    def latest_step(self) -> ProofStep:
        return self.steps[-1]

    @property
    def done(self) -> bool:
        return self.proposition.proved

    @property
    def step(self) -> int:
        return len(self.steps)
    
    
    def save(self): # save at the end of every programming step
        d = {
            'path': self.path,
            'proposition': self.proposition.to_dict(),
            'dialog': self.dialog.to_dict(),
            'steps': [s.to_dict() for s in self.steps],
            'jupyter_session': self.jupyter_session.to_dict(),
            'proxy': self.proxy.to_dict(),
            'terminate_notebook': self.terminate_notebook
        }
        U.save_json(U.pjoin(self.path, 'state.json'), d)

    @classmethod
    def load(cls, path: str, log_base: LogBase):
        d = U.load_json(U.pjoin(path, 'state.json'))
        proxy = Proxy.from_dict(d['proxy'])
        proposition = Proposition.from_dict(d['proposition'], proxy)
        return cls(
            path=d['path'],
            proposition=proposition, 
            dialog=Dialog.from_dict(d['dialog'], log_base, PR), 
            steps=[ProofStep.from_dict(s) for s in d['steps']],
            jupyter_session=JupyterSession.from_dict(d['jupyter_session']),
            proxy=proposition.proxy,
            terminate_notebook=d['terminate_notebook']
        )



class Prover(AgentBase): # programming as proof
    agent_type: AgentType = AgentType.PROVER
    agent_group: List[str] = ['prover']

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream):
        super().__init__(config, ckpt_dir, stream)
        self.prover_agent: Agent = self.agents['prover']
        self.prompts = Prompts('prover') # for reuse
        self.prove_configs = config['prove_configs']
        self.sandbox: JupyterSandbox = JupyterSandbox(config, path=U.pjoin(self.ckpt_dir), verbose=False)
        self.truncate_dialog = self.prove_configs['truncate_dialog']

    def _add_noise(self, p_true: float) -> float:
        if self.prove_configs['noise_level'] > 0:
            p_true = p_true + np.random.normal(0, self.prove_configs['noise_level'])
            p_true = np.clip(p_true, 0, 1)
        return p_true

    def _programming(self, state: ProofState, **kwargs) -> ProofState:
        self.st.write('Programming...')
        debug_max_retries = self.prove_configs['debug_max_retries']

        prompt_args = {}
        prompt_path = 'initial_programming' if state.step == 0 else 'next_programming'
        _programming_prompt = self.prompts(prompt_path)
        _programming_prompt.link_function('retrieve_api_doc', state.proxy.retrieve_api_docs)
        message = self.prover_agent.send_message(
            state.dialog, _programming_prompt, prompt_args
        )
        _dialog = state.dialog.fork() # stop by the request to the agent
        with self.st.status(f'Querying the agent for programming...', expanded=False):
            self.st.write(f'{message.content}')
        with self.st.status(f'Agent is programming...', expanded=True):
            initial_response, _dialog, interrupts = self.prover_agent.call(_dialog)
            parsed = initial_response.parsed
            content = parsed['raw']
            cells = parsed['cells']
            terminate_notebook = parsed['terminate_notebook']
            warnings = parsed['warnings']
            if terminate_notebook and state.step == 0:
                terminate_notebook = False
                warnings.append('You are not allowed to terminate the notebook at the first step, the terminate_notebook flag is ignored')
            self.st.write(f'{content}')
            if warnings:
                self.st.write(f'### Warnings:')
                for warning in warnings:
                    self.st.write(f'{warning}')
                warnings_message = self.prover_agent.send_message(
                    _dialog, self.prompts('warnings_message'), {'warnings': '\n'.join(warnings)}
                )
            if terminate_notebook: # parser will set it to false if there are warnings or python cells
                self.st.write(f'### Terminating the notebook')
            _dialog.overview(remove_tail=True, stream=self.st)

        # Debugging
        if cells:
            passed, _dialog = state.new_step(cells, _dialog)
            initial_error_info = state.latest_program.error_info
            if not passed:
                for i in range(debug_max_retries):
                    U.cprint(f'Debugging... attempt {i+1} of {debug_max_retries}', 'y')
                    prompt_args = {
                        'error_info': state.latest_program.error_info,
                    }
                    message = self.prover_agent.send_message(
                        _dialog, self.prompts('debugging'), prompt_args
                    )
                    with self.st.expander(f'Buggy Code', expanded=True):
                        self.st.write(f'{message.content}')
                    with self.st.status(f'Detected errors in the code, debugging... attempt {i} of {debug_max_retries}', expanded=True):
                        _response, _dialog, _ = self.prover_agent.call(_dialog)
                        parsed = _response.parsed
                        content = parsed['raw']
                        code = parsed['xml_tags']['python_cell']
                        self.st.write(f'{content}')
                        _dialog.overview(remove_tail=True, stream=self.st)
                    passed, _dialog = state.debug(code, _dialog)
                    if passed:
                        U.cprint(f'Debugging passed', 'g')
                        break
            if not passed:
                # state.revert() # step failed, revert to the previous step
                raise ValueError("Failed to program the code")
            
        # step done
        state.step_done()
        if self.truncate_dialog:
            state.dialog.append(initial_response)
            if warnings:
                state.dialog.append(warnings_message)
            if interrupts:
                function_calls = [f'{f}' for f in interrupts if f.success]
                self.prover_agent.send_message(
                    state.dialog, self.prompts('function_calls_message'), 
                    {'function_calls': '\n\n'.join(function_calls)}
                )
            if cells and initial_error_info is not None: # if there are debug, show the summary
                prompt_args = {
                    'error_info': initial_error_info,
                    'program_updates': state.latest_step.program_updates,
                }
                self.prover_agent.send_message(
                    state.dialog, self.prompts('debugging_truncate'), prompt_args
                )
            state.dialog = state.send_latest(state.dialog)
        else:
            state.dialog = _dialog # if not truncate, we need to keep the entire dialog

        state.terminate_notebook = terminate_notebook
        return state

    def _concluding(self, state: ProofState) -> Tuple[Proof, ProofState]:
        message = self.prover_agent.send_message(
            state.dialog,
            self.prompts('concluding'),
        )
        self.st.write(f'{message.content}')
        with self.st.status(f'Concluding...', expanded=False):
            _response, state.dialog, _ = self.prover_agent.call(state.dialog)
            parsed = _response.parsed
            self.st.write(f'{parsed["raw"]}')
            # self.prover_agent.send_message(
            #     state.dialog, self.prompts('classify')
            # )
            # p_true, state.dialog = self.prover_agent.binary_classify(state.dialog)
            self.st.markdown(f'**Truth Value: {parsed["p_true"]:.2f}**\n**Key Factor: {parsed["key_factor"]}**')
            state.dialog.overview(remove_tail=True, stream=self.st)
        proof = Proof(
            proof=parsed['proof'],
            p_true=parsed['p_true'],
            key_factor=parsed['key_factor'],
        )
        state.save()
        return proof, state
    
    def clean_session(self, session_path: str):
        # move everything to a backup folder and make it empty
        _backup_path = U.pjoin(session_path, 'backup')
        U.mkdirs(_backup_path)
        n_backups = len(os.listdir(_backup_path))
        _backup_dir = U.pjoin(_backup_path, f'backup-{n_backups}')
        _files = os.listdir(session_path)
        if 'backup' in _files:
            _files.remove('backup')
        if len(_files) == 0:
            return
        U.mkdirs(_backup_dir) # create the backup folder
        for _file in _files:
            shutil.move(U.pjoin(session_path, _file), U.pjoin(_backup_dir, _file))

    def init_session(self, proposition: Proposition, load_state: bool = False) -> ProofState:
        proxy = proposition.proxy
        assert proxy is not None, "Proxy is not set in the proposition, it serves as open-ended context"
        session_path = U.pjoin(proposition.ckpt_folder, 'session')
        if U.pexists(U.pjoin(session_path, 'state.json')):
            if load_state: # FIXME: this is not safe, cannot handle many cases like unexecpted exit
                state = ProofState.load(session_path, self._log_base)
                self.st.write(f'Loaded checkpoint: {session_path}\nNotebook dir: {state.notebook_dir}\nCkpt dir: {state.path}')
                return state
            else:
                U.cprint(f'Checkpoint found, but load_state is False, skipping', 'y')
        if load_state:
            _get_sess_fn = self.sandbox.get_session
        else:
            _get_sess_fn = self.sandbox.new_session
            self.clean_session(session_path)
        dialog = self.prover_agent.init_dialog({'api_directory': proxy.api_directory})
        self.prover_agent.send_message(
            dialog, self.prompts('request'), {'proposition': proposition.prompt, 'date': proposition.date}
        )
        metadata = {
            'proxy': {
                'activate_proxies': proxy.activate_proxies,
                'cutoff_date': proxy.cutoff_date,
                'deploy_mode': proxy.deploy_mode,
            }
        }
        session_name = proposition.session_name
        state = ProofState(path=session_path, proposition=proposition, dialog=dialog, 
            jupyter_session=_get_sess_fn(session_name, metadata=metadata, path=session_path), proxy=proxy)
        self.st.write(f'New session initialized: {session_name}\nNotebook dir: {state.notebook_dir}\nCkpt dir: {state.path}') 
        return state

    def prove(self, proposition:Proposition) -> Proposition:
        """
        Proposition -> ([Planning -> Programming] x N -> Concluding) -> proof, each prove corresponds to a proposition, one notebook
        Prove is atomic, self-contained, functional, no side effect, it maintain an independent dialog internally, accepts a proposition and returns a proof
        Proxy is used for checking and building for the notebook sandbox, the prove programming environment
        
        ID is for checkpointing, if not provided, a new session will be initialized
        """
        max_proof_steps = self.prove_configs['max_proof_steps']
        state = self.init_session(proposition)
        with self.st.expander('System Prompt Loaded', expanded=False):
            self.st.write(f'{state.dialog.system.content}')
        if state.done:
            return state.proposition
        for i in range(max_proof_steps):
            # TODO: handle checkpointing 
            if i < state.step:
                continue
            state = self._programming(state)
            if state.terminate_notebook:
                break
        if not state.terminate_notebook:
            self.prover_agent.send_message(
                state.dialog, self.prompts('max_step_reached')
            )
        proof, state = self._concluding(state)
        proof.p_true = self._add_noise(proof.p_true)
        state.proposition.prove(proof)
        return state.proposition
    
    def call(self, query: Query, **kwargs) -> Report:
        assert self.ckpt_dir is not None, "ckpt_dir is not set"
        self.st.write(f'{query.prompt}')

        decisions = []
        _options = query.task.get_options()
        # U.cprint(f'Analyzing {len(_options)} options for {query.qid}', color='y')
        for idx in range(len(_options)):
            proposition = query.init_analysis(idx)
            proposition = self.prove(proposition) 
            decisions.append(Decision.from_proposition(idx, proposition))

        report = Report(task=query.task, _decisions=decisions, threshold=self.threshold)
        with self.st.expander('Prediction Overview', expanded=True):
            self.st.code(f'{report}')
        return report


class AsyncProver(Prover):
    agent_type: AgentType = AgentType.ASYNC_PROVER
    is_async: bool = True # override the base class

    async def async_prove(self, proposition: Proposition) -> Proposition:
        loop = asyncio.get_running_loop()
        proposition = await loop.run_in_executor(None, self.prove, proposition) # Runs _prove_sync in a thread pool
        return proposition



class AsyncVanillaProver(AsyncProver):
    agent_type: AgentType = AgentType.ASYNC_VANILLA_PROVER
    agent_group: List[str] = ['vanilla_prover', 'prover'] # prover is just for simplicity

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream, **kwargs):
        super().__init__(config, ckpt_dir, stream)
        self.agent: Agent = self.agents['vanilla_prover']
        self.prompts = Prompts('vanilla')

    def prove(self, proposition: Proposition) -> Proposition:
        """
        Proposition -> ([Planning -> Programming] x N -> Concluding) -> proof, each prove corresponds to a proposition, one notebook
        Prove is atomic, self-contained, functional, no side effect, it maintain an independent dialog internally, accepts a proposition and returns a proof
        Proxy is used for checking and building for the notebook sandbox, the prove programming environment
        
        ID is for checkpointing, if not provided, a new session will be initialized
        """
        dialog = self.agent.init_dialog()
        message = self.agent.send_message(
            dialog, self.prompts('vanilla_prover_task'), 
            {'proposition': proposition.prompt, 'date': proposition.date}
        )
        self.st.write(f'{message.content}')
        with self.st.status(f'Concluding...', expanded=False):
            _response, dialog, _ = self.agent.call(dialog)
            parsed = _response.parsed
            self.st.write(f'{parsed["raw"]}')
            self.st.markdown(f'**Truth Value: {parsed["p_true"]:.2f}**\n**Key Factor: {parsed["key_factor"]}**')
            dialog.overview(remove_tail=True, stream=self.st)
        proof = Proof(
            proof=parsed['proof'],
            p_true=parsed['p_true'],
            key_factor=parsed['key_factor'],
        )
        proof.p_true = self._add_noise(proof.p_true)
        proposition.prove(proof)
        proposition.save_dialog(dialog, 'vanilla_prove')
        return proposition



class DeepResearchProver(Prover):
    agent_type: AgentType = AgentType.DEEP_RESEARCH_PROVER
    is_async: bool = False # override the base class

    def prove(self, proposition: Proposition) -> Proposition:
        model = self.prove_configs['deep_research_model']
        response, parsed = query_dr(proposition, DeepResearchModels(model))
        proof = Proof(
            proof=parsed['proof'],
            p_true=parsed['p_true'],
            key_factor=parsed['key_factor'],
        )
        proof.p_true = self._add_noise(proof.p_true)
        proposition.prove(proof)
        proposition.save_object(response.model_dump(), 'dr_response')
        return proposition

class AsyncDeepResearchProver(DeepResearchProver):
    agent_type: AgentType = AgentType.ASYNC_DEEP_RESEARCH_PROVER
    is_async: bool = True # override the base class
    
    async def async_prove(self, proposition: Proposition) -> Proposition:
        loop = asyncio.get_running_loop()
        proposition = await loop.run_in_executor(None, self.prove, proposition) # Runs _prove_sync in a thread pool
        return proposition


@dataclass
class AnalyticaState:
    path: str
    proposition: Proposition
    dialog: Dialog
    jupyter_session: JupyterSession
    proxy: Proxy
    terminate_notebook: bool = False
    steps: List[ProofStep] = field(default_factory=list)



class Analytica(AsyncAgentBase): # proof tree
    '''
    It uses the pure function of Proveer to build a proof tree
    '''
    agent_type: AgentType = AgentType.ANALYTICA
    agent_group: List[str] = ['analyzer', 'synthesizer']

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream):
        super().__init__(config, ckpt_dir, stream)
        prove_configs = config['prove_configs']
        prover_type = ProverType(prove_configs.get('prover_type', 'programming'))
        if prover_type == ProverType.VANILLA:
            # U.cprint(f'Using vanilla prover', 'y')
            self._prover = AsyncVanillaProver(config, ckpt_dir, stream)
        elif prover_type == ProverType.DEEP_RESEARCH:
            # U.cprint(f'Using deep research prover', 'y')
            self._prover = AsyncDeepResearchProver(config, ckpt_dir, stream)
        elif prover_type == ProverType.PROGRAMMING:
            # U.cprint(f'Using programming prover', 'y')
            self._prover = AsyncProver(config, ckpt_dir, stream)
        else:
            raise ValueError(f'Invalid prover type: {prover_type}')
        self._async_prove = self._prover.async_prove
        self.analyzer: Agent = self.agents.get('analyzer', None)
        self.synthesizer: Agent = self.agents.get('synthesizer', None)
        self.prompts = Prompts('analytica')
        self.max_n_leaves = prove_configs['max_n_leaves']
        self.max_concurrent_prove = prove_configs['max_concurrent_prove']
        self._prove_semaphore = asyncio.Semaphore(self.max_concurrent_prove)  # Limit to 5 concurrent calls
        self.max_proof_retries = prove_configs['max_proof_retries']

    def set_st(self, session_name: str):
        self.st = StreamWrapper(self._stream, self._log_base, session_name)
        self._prover.set_st(session_name)

    def restore_st(self):
        self.st = None
        self._prover.restore_st()   

    def silent(self):
        self._stream = PrintSystem(silent=True)
        self._prover.silent()

    def restore(self):
        self._stream = self._stream_backup  
        self._prover.restore()   


    def run_synthesize(self, proposition: Proposition) -> Union[Proposition, Tuple[Proposition, Dialog]]:
        dialog = self.synthesizer.init_dialog()
        with self.st.status(f'Requesting synthesis for {proposition.sentence}', expanded=False):
            message = self.synthesizer.send_message(
                dialog, self.prompts('synthesize'), {'proposition': proposition.json, 'date': proposition.date})
            self.st.write(f'{message.content}')
        with self.st.status(f'Synthesizing {proposition.sentence}...', expanded=False):
            _response, dialog, _ = self.synthesizer.call(dialog)
            parsed = _response.parsed
            self.st.write(f'{parsed["raw"]}')
            proof = Proof(
                proof=parsed['proof'],
                p_true=parsed['p_true'],
                key_factor=parsed['key_factor'],
            )
            proposition.prove(proof)
        return proposition, dialog

    def _synthesize(self, proposition: Proposition, return_dialog: bool = False, resynthesize_key: str = None) -> Union[Proposition, Tuple[Proposition, Dialog]]:
        _is_resynthesize = resynthesize_key is not None
        assert isinstance(proposition, Proposition)
        assert not proposition.is_leaf
        assert not proposition.proved or _is_resynthesize
        assert proposition.children is not None
        assert proposition.children.proved, "Children are not fully proved"
        if _is_resynthesize:
            assert isinstance(resynthesize_key, str), "resynthesize_key must be a string"
            assert resynthesize_key != '', "resynthesize_key must be a non-empty string"
        U.cprint(f'Start synthesizing: {proposition.sentence}', 'y')
        proposition, dialog = self.run_synthesize(proposition)
        U.cprint(f'Synthesis done: {proposition.sentence}', 'g')   
        ext = f'_{resynthesize_key}' if _is_resynthesize else ''
        proposition.save_dialog(dialog, f'synthesis{ext}')
        return (proposition, dialog) if return_dialog else proposition    
        
    async def _async_synthesize(self, proposition: Proposition, return_dialog: bool = False, resynthesize_key: str = None) -> Proposition:
        loop = asyncio.get_running_loop()
        proposition = await loop.run_in_executor(None, self._synthesize, proposition, return_dialog, resynthesize_key)
        return proposition

    async def synthesize(self, proposition: Proposition, resynthesize_key: str = None) -> Proposition:
        # traverse the tree and gradually synthesize the proof of the proposition backward
        _is_resynthesize = resynthesize_key is not None
        if _is_resynthesize:
            assert isinstance(resynthesize_key, str), "resynthesize_key must be a string"
            assert resynthesize_key != '', "resynthesize_key must be a non-empty string"
        _sentence = proposition.sentence + '(leaf)' if proposition.is_leaf else proposition.sentence + '(non-leaf)'
        if proposition.is_leaf:
            if _is_resynthesize:
                assert proposition.proved, "Proposition leaf is not proved, cannot resynthesize"
                return proposition
            async with self._prove_semaphore:  # Acquire semaphore before proving
                U.cprint(f'Start proving: {proposition.pid} {_sentence}', 'y')
                for i in range(self.max_proof_retries):
                    try:
                        if not proposition.proved:
                            proposition = await self._async_prove(proposition)
                            proposition.save() # just need to save each individual proposition
                        return proposition
                    except Exception as e:
                        proposition.reset()
                        U.cprint(f'Proving failed for {proposition.pid}: {e}, retry {i+1}/{self.max_proof_retries}...', 'r')
                        continue
                raise Exception(f'Proving failed after {self.max_proof_retries} retries for {proposition.pid}: {proposition.sentence}')
        else:
            U.cprint(f'Open synthesizing: {proposition.pid} {_sentence}', 'w')
            tasks = []
            for child in proposition.children.propositions.values():
                tasks.append(self.synthesize(child, resynthesize_key=resynthesize_key))
            await asyncio.gather(*tasks) # block in "backward" direction
            if not proposition.proved or _is_resynthesize:
                if _is_resynthesize and proposition.resynthesized(resynthesize_key):
                    return proposition.load_key(resynthesize_key=resynthesize_key)
                else:
                    proposition = await self._async_synthesize(proposition, return_dialog=False, resynthesize_key=resynthesize_key) # return the task and block in the await above
                    ext = f'_{resynthesize_key}' if _is_resynthesize else ''
                    proposition.save(ext=ext)
            return proposition

    def _analyze(self, proposition: Proposition, return_dialog: bool = False) -> Union[Proposition, Tuple[Proposition, Dialog]]:
        # directly return the whole proposition tree        
        dialog = self.analyzer.init_dialog()
        # U.cprint(f'Start analyzing: {proposition.pid} {proposition.sentence}', 'y')
        for step in range(100):
            if proposition.n_leaves >= self.max_n_leaves:
                break
            with self.st.status(f'Analyzing step {step}', expanded=False):
                if step == 0:
                    message = self.analyzer.send_message(
                        dialog, self.prompts('analyze'), {'proposition': proposition.prompt, 'date': proposition.date})
                else:
                    message = self.analyzer.send_message(dialog, self.prompts('continue_analyze'))
                self.st.write(f'{message.content}')
            with self.st.status(f'Analyzer is analyzing...', expanded=False):
                _response, dialog, _ = self.analyzer.call(dialog, parser_args={'current_nodes': proposition.all_nodes})
                parsed = _response.parsed
                content = parsed['raw']
                self.st.write(f'{content}')
                root, nodes, reasoning, EOA = parsed['root'], parsed['nodes'], parsed['reasoning'], parsed['END_OF_ANALYSIS']
                proposition.add_reasoning(f'# Analysis step {step}\n\n{reasoning}')
                if EOA:
                    break
                else:
                    proposition.add_children(nodes, root)
        proposition.analyze_done(dialog)
        return (proposition, dialog) if return_dialog else proposition

    async def analyze(self, proposition: Proposition, return_dialog: bool = False) -> Union[Proposition, Tuple[Proposition, Dialog]]:
        loop = asyncio.get_running_loop()
        # Run the blocking `analyze` function in a separate thread
        return await loop.run_in_executor(None, self._analyze, proposition, return_dialog)
    
    async def async_prove(self, proposition: Proposition, resynthesize_key: str = None) -> Proposition:
        assert isinstance(proposition, Proposition)
        assert proposition.is_root
        if not proposition.proved:
            if not proposition.analyzed:
                proposition = await self.analyze(proposition)
            proposition = await self.synthesize(proposition, resynthesize_key=resynthesize_key)
        return proposition

    async def call(self, query: Query, resynthesize_key: str = None, **kwargs) -> Report:
        assert self.ckpt_dir is not None, "ckpt_dir is not set"
        self.st.write(f'{query.prompt}')

        tasks = []
        _options = query.task.get_options()
        # U.cprint(f'Analyzing {len(_options)} options', color='y')
        for idx in range(len(_options)):
            proposition = query.init_analysis(idx)
            tasks.append(self.async_prove(proposition, resynthesize_key=resynthesize_key))
        propositions = await asyncio.gather(*tasks)
        decisions = [Decision.from_proposition(idx, proposition) for idx, proposition in enumerate(propositions)]

        report = Report(task=query.task, _decisions=decisions, threshold=self.threshold)
        with self.st.expander('Prediction Overview', expanded=True):
            self.st.code(f'{report}')
        return report

class AnalyticaLinear(Analytica):
    agent_type: AgentType = AgentType.ANALYTICA_LINEAR
    agent_group: List[str] = ['linear_analyzer', 'linear_synthesizer']

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream):
        super().__init__(config, ckpt_dir, stream)
        self.analyzer: Agent = self.agents['linear_analyzer']
        self.synthesizer: Agent = self.agents['linear_synthesizer']
        self.abs_intercept_max = config['prove_configs']['abs_intercept_max']
        assert self.abs_intercept_max > 0, "intercept_max_abs must be positive"
        
    def silent(self):
        self._stream = PrintSystem(silent=True)
        self._prover.silent()

    def restore(self):
        self._stream = self._stream_backup  
        self._prover.restore()   

    def run_synthesize(self, proposition: Proposition) -> Union[Proposition, Tuple[Proposition, Dialog]]:
        dialog = self.synthesizer.init_dialog({'abs_intercept_max': self.abs_intercept_max})
        with self.st.status(f'Requesting synthesis for {proposition.pid} {proposition.sentence}', expanded=False):
            message = self.synthesizer.send_message(
                dialog, self.prompts('linear_synthesize'), {'proposition': proposition.json, 'date': proposition.date})
            self.st.write(f'{message.content}')
        with self.st.status(f'Synthesizing {proposition.pid} {proposition.sentence}...', expanded=False):
            _response, dialog, _ = self.synthesizer.call(dialog, 
                parser_args={'probabilities': proposition.children_probabilities, 'abs_intercept_max': self.abs_intercept_max})
            parsed = _response.parsed
            self.st.write(f'{parsed["raw"]}')
            proof = Proof(
                proof=parsed['proof'],
                p_true=parsed['p_true'],
                key_factor=parsed['key_factor'],
            )
            proposition.prove(proof)
            proposition.set_beta(parsed['beta'])
        return proposition, dialog
 


class AnalyticaLogical(Analytica):
    agent_type: AgentType = AgentType.ANALYTICA_LOGICAL
    agent_group: List[str] = ['logical_analyzer', 'logical_synthesizer']

    def __init__(self, config: Dict[str, Any], ckpt_dir: str, stream):
        super().__init__(config, ckpt_dir, stream)
        self.analyzer: Agent = self.agents['logical_analyzer']
        self.synthesizer: Agent = self.agents['logical_synthesizer']
        
    def silent(self):
        self._stream = PrintSystem(silent=True)
        self._prover.silent()

    def restore(self):
        self._stream = self._stream_backup  
        self._prover.restore()   

    def run_synthesize(self, proposition: Proposition) -> Union[Proposition, Tuple[Proposition, Dialog]]:
        dialog = self.synthesizer.init_dialog()
        with self.st.status(f'Requesting synthesis for {proposition.pid} {proposition.sentence}', expanded=False):
            message = self.synthesizer.send_message(
                dialog, self.prompts('logical_synthesize'), {'proposition': proposition.json, 'date': proposition.date})
            self.st.write(f'{message.content}')
        with self.st.status(f'Synthesizing {proposition.pid} {proposition.sentence}...', expanded=False):
            _response, dialog, _ = self.synthesizer.call(dialog, 
                parser_args={'probabilities': proposition.children_probabilities})
            parsed = _response.parsed
            self.st.write(f'{parsed["raw"]}')
            proof = Proof(
                proof=parsed['proof'],
                p_true=parsed['p_true'],
                key_factor=parsed['key_factor'],
            )
            proposition.prove(proof)
            proposition.set_logic(parsed['logic'])
        return proposition, dialog



AGENT_REGISTRY: Dict[AgentType, AgentBase] = {}

# add all classes in this file to AGENT_REGISTRY if it is a subclass or subsubclass or subsubsubclass... of AgentBase

# traverse all classes in this file

def traverse_agentbase_classes(cls):
    for subcls in cls.__subclasses__():
        # check if subcls is a subclass of AgentBase
        if issubclass(subcls, AgentBase):
            assert subcls.agent_type not in AGENT_REGISTRY, f"Agent {subcls.agent_type} already exists"
            AGENT_REGISTRY[subcls.agent_type] = subcls
        traverse_agentbase_classes(subcls)

traverse_agentbase_classes(AgentBase)


U.cprint(f'{len(AGENT_REGISTRY)} agents registered: {", ".join([str(agent_type) for agent_type in AGENT_REGISTRY.keys()])}', 'g')

def build_agent(config: Dict[str, Any], ckpt_dir: str, stream, agent_type: AgentType = None) -> AgentBase:
    assert 'log_dir' in config, "log_dir is not set"
    if agent_type is None:
        agent_type = AgentType(config['agent_type'])
    elif isinstance(agent_type, str):
        agent_type = AgentType(agent_type)
    return AGENT_REGISTRY[agent_type](config, ckpt_dir, stream)





