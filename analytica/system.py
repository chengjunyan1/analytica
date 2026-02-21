from analytica.proxy.re import Proxy,Maker,Query
from analytica.agent.ae import build_agent, Report, AgentType, Result, PrintSystem
import analytica.utils as U
from typing import List, Dict
import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from dataclasses import dataclass, field
import random
import traceback
from enum import Enum
from tqdm import tqdm
import concurrent.futures
import time

U.load_env()


class TickerState(Enum):
    NOT_TESTED = 'not_tested'
    TESTED = 'tested'

@dataclass
class Ckpt:
    ckpt_dir: str
    tickers: List[str]
    cross_selection: bool = False
    reports: Dict[str, Report] = field(default_factory=dict)

    def __post_init__(self):
        if self.cross_selection:
            self.tickers = [self.xticker_to_str(t) for t in self.tickers]

        def _load_single_report(ticker):
            return ticker, self.load_report(ticker)

        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            futures = {
                executor.submit(_load_single_report, ticker): ticker
                for ticker in self.tickers
                if ticker not in self.reports
            }
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Loading', position=0, leave=True):
                ticker, report = f.result()
                self.reports[ticker] = report

        U.mkdirs(self.ckpt_dir)
        self.save()

    def get_result(self):
        return Result([self.reports[t] for t in self.tickers if self.reports[t] is not None])

    def xticker_to_str(self, xticker: List[str]):
        if self.cross_selection and isinstance(xticker, list):
            return '_vs_'.join(xticker)
        else:
            return xticker

    def str_to_xticker(self, xticker: str):
        if self.cross_selection:
            return xticker.split('_vs_')
        else:
            return xticker

    def get_tickers(self):
        if self.cross_selection:
            return [self.str_to_xticker(t) for t in self.tickers]
        else:
            return self.tickers
        
    def get_ticker_dir(self, ticker: str | List[str]):
        if self.cross_selection and isinstance(ticker, list):
            ticker = self.xticker_to_str(ticker)
        ticker_dir = U.pjoin(self.ckpt_dir, 'tickers', ticker)
        return ticker_dir

    def save_report(self, ticker: str | List[str], report: Report):
        if self.cross_selection and isinstance(ticker, list):
            ticker = self.xticker_to_str(ticker)
        _ticker_dir = self.get_ticker_dir(ticker)
        U.mkdirs(_ticker_dir)
        report.save(U.pjoin(_ticker_dir, f'report.json'))
        self.reports[ticker] = report
        state = self.load_state(ticker)
        state['tested'] = True
        self.save_state(ticker, state)

    def load_report(self, ticker: str | List[str]):
        _path = U.pjoin(self.get_ticker_dir(ticker), f'report.json')
        if U.pexists(_path):
            return Report.load(_path, ckpt_dir=self.ckpt_dir)
        else:
            return None
        
    def load_state(self, ticker: str | List[str]):
        _path = U.pjoin(self.get_ticker_dir(ticker), 'state.json')
        if U.pexists(_path):
            return U.load_json(_path)
        else: 
            return {}
        
    def save_state(self, ticker: str | List[str], state: dict):
        _ticker_dir = self.get_ticker_dir(ticker)
        U.mkdirs(_ticker_dir)
        _path = U.pjoin(_ticker_dir, 'state.json')
        U.save_json(_path, state)
        
    def ticker_tested(self, ticker: str | List[str]):
        if self.cross_selection and isinstance(ticker, list):
            ticker = self.xticker_to_str(ticker)
        return ticker in self.reports and self.reports[ticker] is not None
    
    @property
    def n_evaluated(self):
        return len([t for t in self.tickers if self.ticker_tested(t)])
    
    @property
    def n_tickers(self):
        return len(self.tickers)
    
    def start_test(self, ticker: str | List[str]): # resume or new test
        if self.cross_selection and isinstance(ticker, list):
            ticker = self.xticker_to_str(ticker)
        state = self.load_state(ticker)
        if 'qid' in state and state['qid'] is not None:
            return state['qid']
        # _qid = U.remove_ansi_codes(ticker)
        # qid = f'{_qid}_{U.dt_now_str()}_{U.random_str(4)}'
        qid = ticker # for simplicity
        self.save_state(ticker, {'qid': qid, 'tested': False})
    
    @classmethod
    def from_dict(cls, d: dict, ckpt_dir: str = None):
        d['ckpt_dir'] = ckpt_dir if ckpt_dir else d['ckpt_dir']
        return cls(**d)
    
    def to_dict(self):
        return {
            'tickers': self.tickers,
            'ckpt_dir': self.ckpt_dir,
            'cross_selection': self.cross_selection,
        }
    
    def save(self):
        U.save_json(U.pjoin(self.ckpt_dir, 'state.json'), self.to_dict())




class SystemBase:
    """
    Analytical System Base
    
     - Maker: Make the task
     - Proxy: Proxy of tools
     - Agent: Agent to solve the task
    """
    def __init__(self,config, stream, exp_name=None, deploy_mode=False):
        if exp_name is None:
            exp_name = U.dt_now_str() + '_' + U.random_str(6)
        # set log dir to pass to agent
        self.config = config
        self.set_path(exp_name)
        self.maker = Maker(config, deploy_mode, self.ckpt_dir)
        self.agent = build_agent(config, self.ckpt_dir, stream)
        self._set_exp_name(exp_name)
        self.st = stream
        self.__exp_name_eval_backup = exp_name # for cross selection


    @property
    def agent_type(self):
        return self.agent.agent_type
    
    def set_path(self, exp_name: str):
        self.ckpt_dir = U.pjoin(os.getenv('CKPT_DIR'), self.config['name'], exp_name)
        self.config['log_dir'] = U.pjoin(os.getenv('LOG_DIR'), self.config['name'], exp_name)
    
    def _set_exp_name(self, exp_name: str):
        self.exp_name = exp_name
        self.set_path(exp_name)
        self.maker.ckpt_dir = self.ckpt_dir
        self.agent.ckpt_dir = self.ckpt_dir

    def rebuild(self, agent_type: AgentType | str, exp_name: str = None):
        if isinstance(agent_type, str):
            agent_type = AgentType(agent_type)
        if exp_name is not None:
            self._set_exp_name(exp_name)
        self.agent = build_agent(self.config, self.ckpt_dir, self.st, agent_type)
        self.__exp_name_eval_backup = self.exp_name

    def clone_agent(self): # for safe multi-threading
        return build_agent(self.config, self.ckpt_dir, self.st, self.agent.agent_type)
    
    @property
    def deploy_mode(self):
        return self.maker.deploy_mode   

    def deploy(self):
        self.maker.deploy()
    
    def develop(self):
        self.maker.develop()

    def make_query(self, ticker, qid: str = None, **kwargs) -> Query:
        if isinstance(ticker, str) or isinstance(ticker, list):
            query = self.maker.make_query(ticker, qid=qid)
        elif isinstance(ticker, Query):
            query = ticker
        else:
            raise ValueError(f'Invalid ticker: {ticker}')
        return query

    def call(self, ticker, agent = None, qid: str = None, resynthesize_key: str = None, **kwargs) -> Report:
        U.cprint(f'Calling {ticker}...', color='y')
        agent = self.agent if agent is None else agent
        assert not agent.is_async, 'Agent is async, use async_call instead'
        query = self.make_query(ticker, qid=qid)
        report = agent(query, resynthesize_key=resynthesize_key, **kwargs)
        U.cprint(f'Finished {ticker}...', color='g')
        return report
    
    async def async_call(self, ticker, agent = None, qid: str = None, resynthesize_key: str = None, **kwargs) -> Report:
        U.cprint(f'Calling {ticker}...', color='y')
        agent = self.agent if agent is None else agent
        assert agent.is_async, 'Agent is not async, use call instead'
        query = self.make_query(ticker, qid=qid)
        report = await agent(query, resynthesize_key=resynthesize_key, **kwargs)
        U.cprint(f'Finished {ticker}...', color='g')
        return report

    def load_ckpt(self):
        ckpt_path = U.pjoin(self.ckpt_dir, 'state.json')
        if U.pexists(ckpt_path):
            return Ckpt.from_dict(U.load_json(ckpt_path), ckpt_dir=self.ckpt_dir)
        else:
            return None

    def setup_evaluate(
        self,
        cross_selection: bool = False,
        tickers: List[str] = None,  
        show_detail: bool = False,
        n_tickers: int = None,
        max_concurrent: int = 10,
    ) -> Ckpt:
        if tickers is None:
            tickers = self.maker.ticker_sequence if not cross_selection else self.maker.xticker_sequence
            random.shuffle(tickers)
        if n_tickers is not None:
            tickers = tickers[:n_tickers]
        if self.st is None:
            self.st = PrintSystem(silent=not show_detail)
        self.st.write(f'Evaluating {len(tickers)} tickers with exp name: {self.exp_name} in {max_concurrent} threads.') 
        if cross_selection:
            self.__exp_name_eval_backup = self.exp_name
            self._set_exp_name(self.exp_name+'_X')
        else:
            self._set_exp_name(self.__exp_name_eval_backup)
        U.mkdirs(self.ckpt_dir)
        ckpt_path = U.pjoin(self.ckpt_dir, 'state.json')
        if U.pexists(ckpt_path):
            ckpt = Ckpt.from_dict(U.load_json(ckpt_path), ckpt_dir=self.ckpt_dir)
            assert ckpt.cross_selection == cross_selection, f'Cross selection is not consistent with the checkpoint. {ckpt.cross_selection} != {cross_selection}'
            tickers = ckpt.get_tickers()
            U.cprint(f'Resuming from checkpoint: {ckpt.ckpt_dir}, {ckpt.n_evaluated}/{ckpt.n_tickers} tickers evaluated', color='g')
        else:
            ckpt = Ckpt(ckpt_dir=self.ckpt_dir, tickers=tickers, cross_selection=cross_selection)
        return ckpt, tickers
    
    def sequential_evaluate(
            self, 
            cross_selection: bool = False,
            tickers: List[str] = None, 
            show_detail: bool = False,
            n_tickers: int = None,
            resynthesize_key: str = None,
            agent_args: dict = {},
    ) -> Result:
        assert not self.agent.is_async, 'Agent is async, please use async_evaluate instead'
        ckpt, tickers = self.setup_evaluate(cross_selection,tickers,show_detail,n_tickers,1)
        all_reports = []
        
        for ticker in tickers:
            for i in range(self.config['max_query_retry']):
                try:
                    agent = self.clone_agent()
                    if not show_detail:
                        agent.silent()
                    if ckpt.ticker_tested(ticker) and not resynthesize_key:
                        report = ckpt.load_report(ticker)
                    else:
                        qid = ckpt.start_test(ticker)
                        report = self.call(ticker, agent, qid=qid, resynthesize_key=resynthesize_key, **agent_args)
                        ckpt.save_report(ticker, report)
                    all_reports.append(report)
                    break
                except Exception as e:
                    _traceback = traceback.format_exc()
                    if i < self.config['max_query_retry']:
                        self.st.error(f'Error analyzing {ticker}: {str(e)}\n{'-'*100}\n{_traceback}\nRetrying... {i+1}/{self.config['max_query_retry']}')
                        time.sleep(1)
                        
        result = Result(all_reports)
        self.st.write(str(result))
        self._set_exp_name(self.__exp_name_eval_backup)
        return result
    

    def evaluate(
            self, 
            cross_selection: bool = False,
            tickers: List[str] = None, 
            show_detail: bool = False,
            n_tickers: int = None,
            max_concurrent: int = 10,
            resynthesize_key: str = None,
            agent_args: dict = {},
    ) -> Result:
        if resynthesize_key is not None:
            assert 'analytica' in self.agent.agent_type.value, 'Agent is not an analytica agent, it does not support resynthesize'

        assert not self.agent.is_async, 'Agent is async, please use async_evaluate instead'
        ckpt, tickers = self.setup_evaluate(cross_selection,tickers,show_detail,n_tickers,max_concurrent)
        all_reports = []
        my_bar = self.st.progress(0)

        progress_queue = Queue()
        completed = 0

        def call_with_progress(ticker):
            for i in range(self.config['max_query_retry']):
                try:
                    agent = self.clone_agent()
                    if not show_detail:
                        agent.silent()
                    if ckpt.ticker_tested(ticker) and not resynthesize_key:
                        report = ckpt.load_report(ticker)
                    else:
                        qid = ckpt.start_test(ticker)
                        report = self.call(ticker, agent, qid=qid, resynthesize_key=resynthesize_key, **agent_args)
                        ckpt.save_report(ticker, report)
                    progress_queue.put((ticker, report, None, None))
                    break
                except Exception as e:
                    _traceback = traceback.format_exc()
                    if i == self.config['max_query_retry'] - 1:
                        progress_queue.put((ticker, None, e, _traceback))
                    else:
                        self.st.error(f'Error analyzing {ticker}: {str(e)}\n{'-'*100}\n{_traceback}\nRetrying... {i+1}/{self.config['max_query_retry']}')
                        time.sleep(1)
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_ticker = {executor.submit(call_with_progress, ticker): ticker for ticker in tickers}
            
            while completed < len(tickers):
                try:
                    ticker, report, error, _traceback = progress_queue.get(timeout=0.1)
                    completed += 1
                    my_bar.progress(completed/len(tickers), text=f'Evaluating {ticker} ({completed}/{len(tickers)})...')
                    
                    if error is not None:
                        self.st.error(f"Error analyzing {ticker}: {str(error)}\n{'-'*100}\n{_traceback}")
                        continue
                        
                    all_reports.append(report)
                except Empty:
                    # No progress update available, continue waiting
                    continue

        result = Result(all_reports)
        self.st.write(str(result))
        self._set_exp_name(self.__exp_name_eval_backup)
        return result


    async def async_evaluate(
        self,
        cross_selection: bool = False,
        tickers: List[str] = None,
        show_detail: bool = False,
        n_tickers: int = None,
        max_concurrent: int = 5,
        resynthesize_key: str = None,
        agent_args: dict = {},
    ) -> Result:
        if resynthesize_key is not None:
            assert 'analytica' in self.agent.agent_type.value, 'Agent is not an analytica agent, it does not support resynthesize'

        assert self.agent.is_async, 'Agent is not async, please use evaluate instead'
        ckpt, tickers = self.setup_evaluate(cross_selection,tickers,show_detail,n_tickers,max_concurrent)
        my_bar = self.st.progress(0)

        async def call_with_progress(ticker, progress_queue):
            for i in range(self.config['max_query_retry']):
                try:
                    agent = self.clone_agent()
                    if not show_detail:
                        agent.silent()
                    
                    if ckpt.ticker_tested(ticker) and not resynthesize_key:
                        report = ckpt.load_report(ticker)
                    else:
                        qid = ckpt.start_test(ticker)
                        report = await self.async_call(ticker, agent, qid=qid, resynthesize_key=resynthesize_key, **agent_args)
                        ckpt.save_report(ticker, report)
                    
                    await progress_queue.put((ticker, report, None, None))
                    break
                except Exception as e:
                    _traceback = traceback.format_exc()
                    if i == self.config['max_query_retry'] - 1:
                        await progress_queue.put((ticker, None, e, _traceback))
                    else:
                        self.st.error(f'Error analyzing {ticker}: {str(e)}\n{"-"*100}\n{_traceback}\nRetrying... {i+1}/{self.config["max_query_retry"]}')
                        await asyncio.sleep(1)

        progress_queue = asyncio.Queue()
        tasks = [call_with_progress(ticker, progress_queue) for ticker in tickers]
        
        # This is a simple way to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def gather_with_concurrency(tasks):
            async def run_task(task):
                async with semaphore:
                    return await task
            return await asyncio.gather(*(run_task(task) for task in tasks))

        gather_task = asyncio.create_task(gather_with_concurrency(tasks))
        
        completed = 0
        all_reports = []
        while completed < len(tickers):
            try:
                ticker, report, error, _traceback = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                completed += 1
                my_bar.progress(completed/len(tickers), text=f'Evaluating {ticker} ({completed}/{len(tickers)})...')
                
                if error is not None:
                    self.st.error(f"Error analyzing {ticker}: {str(error)}\n{'-'*100}\n{_traceback}")
                    continue
                
                all_reports.append(report)
            except asyncio.TimeoutError:
                if gather_task.done() and progress_queue.empty():
                    break

        await gather_task
        
        result = Result(all_reports)
        self.st.write(str(result))
        self._set_exp_name(self.__exp_name_eval_backup)
        return result




class BasicSystem(SystemBase):
    pass





def build_system(config, exp_name=None, stream=None, deploy_mode=False):
    system_type = config['system_type']
    if system_type == 'basic':
        system = BasicSystem(config, stream, exp_name, deploy_mode)
    else:
        raise ValueError(f'Invalid system type: {system_type}')
    return system

