from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
import datetime as dt
from enum import Enum
import os
import base64
import cairosvg
import shutil
from PIL import Image
import random
from io import BytesIO
from analytica.utils import (remove_ansi_codes, process_html_output, plotly_json_to_base64_png, idx2var,
        replace_special_chars, save_json, load_json, pjoin, pexists, mkdirs, hash_str)

from lllm.sandbox import JupyterCellType, ProgrammingLanguage, JupyterSession 
from lllm.models import Roles
from lllm.llm import Dialog





##### RE 


BASIC_PROXIES = ['fmp','msd','fred','gt','exa']# ,'wa', 'gkg', 'pm', 'kb'

FINANCIAL_CATEGORIES = ['stock','crypto','forex','commodity','index','fund']
PREDMARKET_CATEGORIES = ['kalshi','polymarket']

BASE_COMMISSION = 0.02
RETURN_BOUND = 1 # max +-100% return



class EnumPlus(Enum):
    @classmethod
    def values(cls):
        return [i.value for i in cls.__members__.values()]

class MIMEBaseType(EnumPlus):
    TEXT = 'text'
    IMAGE = 'image'
    APPLICATION = 'application'
    OTHER = 'other'

class ImageType(EnumPlus):
    GIF = 'image/gif'
    JPEG = 'image/jpeg'
    PNG = 'image/png'
    SVG_XML = 'image/svg+xml'
    WEBP = 'image/webp'
    OTHER = 'image/other'

class TextType(EnumPlus):
    PLAIN = 'text/plain'
    MARKDOWN = 'text/markdown'
    HTML = 'text/html' # lengthy
    LATEX = 'text/latex'
    OTHER = 'text/other'

class ApplicationType(EnumPlus):
    JSON = 'application/json'
    WIDGET = 'application/vnd.jupyter.widget-view+json'
    PLOTLY = 'application/vnd.plotly.v1+json'
    OTHER = 'application/other'


def base64_to_image(base64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(base64_str)))

def svg_to_base64_png(svg_string: str) -> str:
    png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
    base64_bytes = base64.b64encode(png_bytes)
    base64_string = base64_bytes.decode('ascii')
    return base64_string



class CellOutputType(Enum):
    STREAM = 'stream'
    DISPLAY_DATA = 'display_data'
    EXECUTE_RESULT = 'execute_result'
    ERROR = 'error'



@dataclass
class MIMEBundleParsed:
    content: str 
    mime_type: str
    is_base64: bool = False

    @classmethod
    def from_data(cls, data: Dict[str, Any]):
        if ImageBundle.check_type(data) is not None:
            return ImageBundle.from_data(data).data
        elif TextBundle.check_type(data) is not None:
            return TextBundle.from_data(data).data
        elif ApplicationBundle.check_type(data) is not None:
            return ApplicationBundle.from_data(data).data
        else:
            raise ValueError(f"Unknown output type: {data.keys()}")

@dataclass
class CellOutput:
    @property
    def type(self):
        raise NotImplementedError
    
    @property
    def name(self):
        raise NotImplementedError
    
    @property
    def is_base64(self):
        raise NotImplementedError


@dataclass
class StreamOutput(CellOutput): # e.g. print, logging, etc.
    _name: str
    _text: str

    def __post_init__(self):
        self.text = remove_ansi_codes(self._text)

    @property
    def name(self):
        return self._name
    
    @property
    def type(self):
        return CellOutputType.STREAM
    
    @property
    def is_base64(self):
        return False
    
    def to_prompt(self, max_length: int = 0, index: int = None) -> str:
        if max_length > 0:
            _prompt = self.text[:max_length] + '...' if len(self.text) > max_length else self.text
        else:
            _prompt = self.text
        if index is not None:
            _prompt = f"[Output {index}: {self.name}]\n\n{_prompt.strip()}"
        else:
            _prompt = f"[{self.name}]\n\n{_prompt.strip()}"
        return _prompt+'\n\n'

@dataclass
class DisplayDataOutput(CellOutput): # e.g. plot, table, etc.
    data: Dict[str, Any]
    metadata: Dict[str, Any]

    def __post_init__(self):
        self.mime_bundle = MIMEBundleParsed.from_data(self.data)

    @property
    def name(self):
        return self.mime_bundle.mime_type

    @property
    def type(self):
        return CellOutputType.DISPLAY_DATA
    
    @property
    def is_base64(self):
        return self.mime_bundle.is_base64

    def to_prompt(self, max_length: int = 0, index: int = None) -> str:
        if self.is_base64:
            return self.mime_bundle.content
        else:
            if max_length > 0:
                _prompt = self.mime_bundle.content[:max_length] + '...' if len(self.mime_bundle.content) > max_length else self.mime_bundle.content
            else:
                _prompt = self.mime_bundle.content
            if index is not None:
                _prompt = f"[Output {index}: {self.name}]\n\n{_prompt.strip()}"
            else:
                _prompt = f"[{self.name}]\n\n{_prompt.strip()}"
            return _prompt+'\n\n'
    

@dataclass
class ExecuteResultOutput(DisplayDataOutput): # e.g. return value, etc.
    execution_count: int

    def __post_init__(self):
        self.mime_bundle = MIMEBundleParsed.from_data(self.data)

    @property
    def type(self):
        return CellOutputType.EXECUTE_RESULT

@dataclass
class ErrorOutput(CellOutput): # e.g. error message, etc.
    ename: str
    evalue: str
    _traceback: str

    def __post_init__(self):
        self.traceback = remove_ansi_codes(self._traceback)

    @property
    def name(self):
        return 'ERROR'
    
    @property
    def type(self):
        return CellOutputType.ERROR
    
    @property
    def is_base64(self):
        return False
    
    def to_prompt(self, max_length: int = 0, index: int = None) -> str:
        if index is not None:
            _prompt = f"[Error {index}: {self.name}]\n\n{self.ename}: {self.evalue}\nTraceback:\n"
        else:
            _prompt = f"[{self.name}]\n\n{self.ename}: {self.evalue}\nTraceback:\n"
        if max_length > 0:
            _prompt += self.traceback[:max_length] + '...' if len(self.traceback) > max_length else self.traceback
        else:
            _prompt += self.traceback
        _prompt = _prompt.strip() + '\n\n'
        return _prompt

@dataclass
class MIMEBundle:
    mime_type: ImageType | TextType | ApplicationType
    text: str 

    @staticmethod
    def check_type(output_data: Dict[str, Any]) -> ImageType | TextType | ApplicationType | None:
        raise NotImplementedError

    @classmethod
    def from_data(cls, output_data: Dict[str, Any]): # output_data is output.data in list of outputs
        raise NotImplementedError

    @property
    def data(self) -> MIMEBundleParsed:
        raise NotImplementedError
    
    @staticmethod
    def get_type(output_data: Dict[str, Any], base_type: str, Types: EnumPlus) -> str:
        if len(output_data.keys()) == 1:
            _type = list(output_data.keys())[0]
        elif len(output_data.keys()) == 2:
            _keys = list(output_data.keys())
            if 'text/plain' not in _keys:
                return None
            _keys.remove('text/plain')
            _type = _keys[0]
        else:
            return None
        if _type.split('/')[0] != base_type:
            return None
        if _type in Types.values():
            _type = Types(_type)
        else:
            _type = Types.OTHER
        return _type

@dataclass
class ImageBundle(MIMEBundle):
    mime_type: ImageType
    text: str
    base64: str = None

    @staticmethod
    def check_type(output_data: Dict[str, Any]) -> ImageType | None: # None means not a ImageBundle
        return ImageBundle.get_type(output_data, 'image', ImageType)

    @classmethod
    def from_data(cls, output_data: Dict[str, Any]): # output_data is output.data in list of outputs
        _type = ImageBundle.check_type(output_data)
        assert _type is not None
        _text = output_data['text/plain']
        try:
            _image = output_data[_type.value]
            if _type == ImageType.SVG_XML:
                _image = svg_to_base64_png(_image)
            _base64 = base64.b64decode(_image)
            _base64 = _image # if parsable, then use the original string
        except:
            _base64 = None
        return cls(mime_type=_type, base64=_base64, text=_text)
    
    @property
    def data(self):
        if self.base64 is not None:
            return MIMEBundleParsed(content=self.base64, is_base64=True, mime_type=self.mime_type.value)
        else:
            return MIMEBundleParsed(content=self.text, is_base64=False, mime_type=self.mime_type.value)

@dataclass
class TextBundle(MIMEBundle):
    mime_type: TextType
    text: str 
    table: str = None
    base64: str = None
    content: str = None

    @staticmethod
    def check_type(output_data: Dict[str, Any]) -> TextType | None:
        return TextBundle.get_type(output_data, 'text', TextType)

    @classmethod
    def from_data(cls, output_data: Dict[str, Any]): # output_data is output.data in list of outputs
        _type = TextBundle.check_type(output_data)
        assert _type is not None
        data = {}
        _text = output_data['text/plain']
        if _type == TextType.PLAIN:
            pass
        elif _type == TextType.HTML:
            _data = process_html_output(output_data['text/html'])
            data['table'] = _data.get('table',None)
            data['base64'] = _data.get('base64',None)
        else:
            try:
                data['content'] = output_data[_type.value]
            except:
                pass
        return cls(mime_type=_type, text=_text, **data)
    
    @property
    def data(self):
        if self.table is not None:
            return MIMEBundleParsed(content=self.table, is_base64=False, mime_type=self.mime_type.value)
        elif self.base64 is not None:
            return MIMEBundleParsed(content=self.base64, is_base64=True, mime_type=self.mime_type.value)
        elif self.content is not None:
            return MIMEBundleParsed(content=self.content, is_base64=False, mime_type=self.mime_type.value)    
        else:
            return MIMEBundleParsed(content=self.text, is_base64=False, mime_type=self.mime_type.value)
        

@dataclass
class ApplicationBundle(MIMEBundle):
    mime_type: ApplicationType
    text: str
    base64: str = None
    content: str = None


    @staticmethod
    def check_type(output_data: Dict[str, Any]) -> ApplicationType | None:
        return ApplicationBundle.get_type(output_data, 'application', ApplicationType)

    @classmethod
    def from_data(cls, output_data: Dict[str, Any]): # output_data is output.data in list of outputs
        _type = ApplicationBundle.check_type(output_data)
        assert _type is not None
        _text = output_data.get('text/plain', None)
        _base64 = None
        _content = None
        if _type == ApplicationType.JSON:
            _content = output_data['application/json']
        elif _type == ApplicationType.PLOTLY:
            _image = output_data['application/vnd.plotly.v1+json']
            _base64 = plotly_json_to_base64_png(_image)
        elif _type == ApplicationType.WIDGET:
            _content = output_data['application/vnd.jupyter.widget-view+json']
        else:
            try:
                output_data.pop('text/plain')
            except:
                pass
            _content = output_data.values()[0] if len(output_data.values()) == 1 else None
        return cls(mime_type=_type, text=_text, base64=_base64, content=_content)


    @property
    def data(self):
        if self.base64 is not None:
            return MIMEBundleParsed(content=self.base64, is_base64=True, mime_type=self.mime_type.value)
        elif self.content is not None:
            return MIMEBundleParsed(content=self.content, is_base64=False, mime_type=self.mime_type.value)
        else:
            return MIMEBundleParsed(content=self.text, is_base64=False, mime_type=self.mime_type.value)




@dataclass
class MIMEOutputs:
    # mime-bundle of data to display in the pager.
    # Must include text/plain.
    # https://jupyter-client.readthedocs.io/en/stable/messaging.html#id4
    data: Dict[str, MIMEBundleParsed]

    @classmethod
    def from_outputs(cls, outputs: List[StreamOutput|DisplayDataOutput|ExecuteResultOutput|ErrorOutput]): 
        _data = []
        for output in outputs:
            if ImageBundle.check_type(output.data) is not None:
                _data.append(ImageBundle.from_data(output.data))
            elif TextBundle.check_type(output.data) is not None:
                _data.append(TextBundle.from_data(output.data))
            elif ApplicationBundle.check_type(output.data) is not None:
                _data.append(ApplicationBundle.from_data(output.data))
            else:
                raise ValueError(f"Unknown output type: {output.type}")
        return cls(_data)
    
    @property
    def is_empty(self):
        return len(self.data) == 0
    


def parse_outputs(outputs: List[Dict[str, Any]]):
    cell_outputs = []
    for d in outputs:
        _type = CellOutputType(d['output_type'])
        if _type == CellOutputType.STREAM:
            cell_outputs.append(StreamOutput(_name=d['name'], _text=''.join(d['text'])))
        elif _type == CellOutputType.DISPLAY_DATA:
            cell_outputs.append(DisplayDataOutput(data=d['data'], metadata=d['metadata']))
        elif _type == CellOutputType.EXECUTE_RESULT:
            cell_outputs.append(ExecuteResultOutput(execution_count=d['execution_count'], data=d['data'], metadata=d['metadata']))
        elif _type == CellOutputType.ERROR:
            cell_outputs.append(ErrorOutput(ename=d['ename'], evalue=d['evalue'], _traceback='\n'.join(d['traceback'])))
        else:
            raise ValueError(f"Unknown output type: {_type}")
    return cell_outputs


@dataclass
class JupyterCellPrompt:
    text: str
    images: Dict[str,str]

    @classmethod
    def from_outputs(cls, outputs: List[StreamOutput|DisplayDataOutput|ExecuteResultOutput|ErrorOutput], max_length: int = 0):
        _strings = []
        _images = {}
        for i, output in enumerate(outputs):
            if output.is_base64:
                name = f'Figure {len(_images)+1}'
                _images[name] = output.to_prompt(max_length=max_length, index=i+1)
                _strings.append(f'[{i+1}: {output.name}]\n\nSee {name}\n\n')
            else:
                _strings.append(output.to_prompt(max_length=max_length, index=i+1))
        if len(_strings) == 0:
            return cls(text='No outputs', images={})
        _splitter = '-'*10 + '\n\n'
        _text = _splitter
        for _string in _strings:
            _text += _string + _splitter
        return cls(text=_text, images=_images)
    
    def send(self, dialog: Dialog, creator: str = 'user') -> Dialog: # images first, then text
        for name, image in self.images.items():
            dialog.send_base64_image(image, caption=name, creator=creator)
        dialog.send_message(self.text, creator=creator, role=Roles.USER)
        return dialog   


@dataclass
class JupyterCell:
    index: int # index in a Program
    source: str
    type: JupyterCellType
    programming_language: ProgrammingLanguage = ProgrammingLanguage.PYTHON
    outputs: List[StreamOutput|DisplayDataOutput|ExecuteResultOutput|ErrorOutput] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    nb_index: int = None # index in a Notebook, set when writing to a Notebook
    

    def to_prompt(self, max_length: int = 1000, summary_length: int = 200) -> JupyterCellPrompt:
        if self.type == JupyterCellType.CODE:
            if summary_length > 0:
                summary = self.source.replace("\n", "\\n")[:summary_length]
                if len(self.source) > summary_length:
                    summary += '...'
                decr = f'Cell {self.nb_index} ({summary}) outputs:\n\n'
            else:
                decr = f'Cell {self.nb_index} (python) outputs:\n\n'
            _prompt = JupyterCellPrompt.from_outputs(self.outputs, max_length=max_length)
            _prompt.text = decr + _prompt.text
            return _prompt
        else:
            summary_length = 100
            summary = self.source.replace("\n", "\\n")[:summary_length]
            if len(self.source) > summary_length:
                summary += '...'
            decr = f'Cell {self.nb_index} ({summary}):\n\nMarkdown cell, no outputs.'
            return JupyterCellPrompt(text=decr, images={})

    def send(self, dialog: Dialog, creator: str = 'user') -> Dialog:
        _prompt = self.to_prompt()
        _prompt.send(dialog, creator=creator)
        return dialog

    @property
    def is_code(self):
        return self.type == JupyterCellType.CODE

    def to_dict(self):
        return {
            'index': self.index,
            'nb_index': self.nb_index,
            'source': self.source,
            'type': self.type.value,
            'programming_language': self.programming_language.value,
            'outputs': self.outputs,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            index=d['index'],
            nb_index=d['nb_index'],
            source=d['source'],
            type=JupyterCellType(d['type']),
            programming_language=ProgrammingLanguage(d['programming_language']),    
            outputs=parse_outputs(d['outputs']),
            metadata=d['metadata'],
        )
    
    @classmethod
    def from_nb_cell(cls, cell: dict, index: int, nb_index: int = None):
        _type = JupyterCellType(cell['cell_type'])
        if _type == JupyterCellType.CODE:
            outputs = parse_outputs(cell['outputs'])
        elif _type == JupyterCellType.MARKDOWN:
            outputs = []
        else:
            raise ValueError(f"Invalid cell type: {_type}")
        return cls(
            index=index,
            source=cell['source'],
            type=_type,
            outputs=outputs,
            metadata=cell['metadata'],
            nb_index=nb_index if nb_index is not None else cell['id'],
        )
    

    

def read_nb_cells(nb: List[dict], nb_indices: List[int] = None) -> List[JupyterCell]:
    if nb_indices is None:
        nb_indices = [None] * len(nb)
    assert len(nb_indices) == len(nb), "The number of cells and indices must be the same"
    cells = []
    for idx, cell in enumerate(nb):
        cells.append(JupyterCell.from_nb_cell(cell, idx, nb_indices[idx]))
    return cells
    

@dataclass
class Failure:
    index: int
    nb_index: int
    error_outputs: List[ErrorOutput|StreamOutput]

    @property
    def error_message(self):
        message = ''
        for output in self.error_outputs:
            if isinstance(output, ErrorOutput):
                message += f"{output.ename}: {output.evalue}\n"
                message += f"{output.traceback}\n"
            elif isinstance(output, StreamOutput):
                message += f"{output.text}\n"
        return message


@dataclass
class Program:
    cells: List[JupyterCell]
    failure: Failure = None
    __written: bool = False

    def send(self, dialog: Dialog, creator: str = 'user') -> Dialog:
        for cell in self.cells:
            cell.send(dialog, creator=creator)
        return dialog

    @property
    def error_info(self) -> str: # where the execution failed
        if self.failure is None:
            return None
        code_short = self.cells[self.failure.index].source[:200].replace("\n", "\\n")
        if len(code_short) < len(self.cells[self.failure.index].source):
            code_short += '...'
        error_short = self.failure.error_message[:200]
        if len(error_short) < len(self.failure.error_message):
            error_short += '...'
        return f'''There are errors when executing Cell {self.failure.index} ({code_short}):
```
{error_short}
```

Please see the detailed error message in the cell outputs. 
'''
    
    def test(self, jupyter_session: JupyterSession) -> bool: # update cells, and return if the execution is successful
        _nb_indices = self.write(jupyter_session)
        # print(f"Testing cells: {_nb_indices}")
        nbidx_to_index = {i: j for i, j in zip(_nb_indices, range(len(_nb_indices)))} # store the mapping between internal index and NB index
        restart = True # self.failure is not None
        failed_nbidx = jupyter_session.run_all_cells(restart=restart) # return the NB index of the failed cell
        # read cell outputs
        _cells = jupyter_session.get_cells(_nb_indices)
        self.cells = read_nb_cells(_cells,_nb_indices) 
        if failed_nbidx is None: # some errors are not reported by NB, check them manually
            failed_nbidx = self.check_errors()
        self.failure = None
        if failed_nbidx is not None:
            assert failed_nbidx in nbidx_to_index, f"Failed to execute the cells before the new cells (should not happen! check the code): {failed_nbidx} not in {nbidx_to_index}" # should not happen
            failed_index = nbidx_to_index[failed_nbidx]
            failed_cell = _cells[failed_index]
            self.failure = Failure(
                index=failed_index, 
                nb_index=failed_nbidx,
                error_outputs=parse_outputs(failed_cell.outputs)
            )
        return self.failure is None
    
    def check_errors(self) -> int:
        for cell in self.cells:
            if cell.type == JupyterCellType.CODE:
                if any(output.type == CellOutputType.ERROR for output in cell.outputs):
                    return cell.nb_index
        return None
    
    def clean(self, jupyter_session: JupyterSession):
        jupyter_session.delete_cells(self.nb_indices)

    def debug(self, code: str, jupyter_session: JupyterSession = None):
        self.cells[self.failure.index].source = code
        jupyter_session.overwrite_cell(self.failure.nb_index, code, JupyterCellType.CODE)
        self.test(jupyter_session)

    @property
    def nb_indices(self) -> List[int]:
        return [i.nb_index for i in self.cells]

    def write(self, jupyter_session: JupyterSession):
        if self.__written:
            return self.nb_indices
        _indices = []
        for cell in self.cells:
            if cell.type == JupyterCellType.CODE:
                _nb_index = jupyter_session.append_code_cell(cell.source)
            elif cell.type == JupyterCellType.MARKDOWN:
                _nb_index = jupyter_session.append_markdown_cell(cell.source)
            else:   
                raise ValueError(f"Invalid cell type: {cell.type}")
            cell.nb_index = _nb_index
            _indices.append(_nb_index)
        self.__written = True
        return _indices
    
    def __str__(self):
        raise
    
    @classmethod
    def from_nb_cells(cls, cells: List[dict]):
        cells = read_nb_cells(cells)
        return cls(cells=cells)
    




def synthesizer_evaluator(program: str, children_variables: Dict[str, float]) -> dict:
    result = {
        'passed': False,
        'error': [],
        'probability': None,
        'source_code': program,
        'function': None,
    }
    # 1. check if the last two lines unchanged
    lines = [i.strip() for i in program.split('\n') if i.strip() != '']
    if not lines[-2].startswith(f'P = {children_variables}'):
        result['error'].append(f"The second last line should be P = {children_variables}, but got {lines[-2]}")
    if not lines[-1].startswith('synthesized_probability = synthesize_probability(P)'):
        result['error'].append(f"The last line should be synthesized_probability = synthesize_probability(P), but got {lines[-1]}")
    # 2. check if the program is valid
    try:
        locals = {}
        exec(program, locals)
        synthesized_probability = locals['synthesized_probability']
        synthesizer_function = locals['synthesize_probability']
        result['function'] = synthesizer_function
        if not callable(synthesizer_function):
            result['error'].append("synthesize_probability is not a function")
        if not isinstance(synthesized_probability, float):
            result['error'].append("synthesized_probability is not a float")
        if synthesized_probability < 0 or synthesized_probability > 1:
            result['error'].append(f"synthesized_probability should be between 0 and 1, but got {synthesized_probability}")
        # check if the synthesized probability is correct
        if synthesized_probability != synthesizer_function(children_variables):
            result['error'].append(f"synthesized_probability is not correct, got {synthesized_probability}, expected {synthesizer_function(children_variables)}")
    except Exception as e:
        result['error'].append(str(e))
    result['passed'] = len(result['error']) == 0
    if result['passed']:
        result['probability'] = synthesized_probability
    return result



@dataclass
class RelationalPropositions:
    propositions: Dict[str, 'Proposition'] = field(default_factory=dict)
    causality: str = None 
    ckpt_dir: str = None
    qid: str = None
    oid: str = None
    beta: Dict[str, float] = field(default_factory=dict)
    logic: Dict[str, Any] = field(default_factory=dict) 

    def append(self, proposition: 'Proposition'):
        child_pid = proposition.pid
        assert isinstance(child_pid, str), "Child pid should be a string"
        assert child_pid not in self.propositions, f"Child pid {child_pid} already exists"
        self.propositions[child_pid] = proposition
        proposition.pid = child_pid
        
    @property
    def proved(self):
        return all(c.proved for c in self.propositions.values())

    @property
    def json(self):
        assert self.proved, "Propositions are not fully proved"
        return {
            'propositions': [
                {
                    'proposition_id': c.pid,
                    'proposition': c.sentence,
                    'p_true': c.proof.p_true,
                    'proof': c.proof.proof,
                    'key_factor': c.proof.key_factor,
                }
                for v, c in self.propositions.items()
            ],
            'causality': self.causality,
        }
    
    @property
    def variables(self):
        assert self.proved, "Propositions are not fully proved"
        return {v: c.proof.p_true for v, c in self.propositions.items()}
        

    def to_dict(self): # only save the ids of the propositions, need to load separately
        return {
            'causality': self.causality,
            'children_pids': list(self.propositions.keys()),
            'qid': self.qid,
            'oid': self.oid,
            'ckpt_dir': self.ckpt_dir,
            'beta': self.beta,
            'logic': self.logic,
        }

    @classmethod
    def from_dict(cls, d: dict, proxy: 'Proxy', ckpt_dir: str = None):
        # assert proxy is not None, 'Proxy is required'
        d['ckpt_dir'] = ckpt_dir if ckpt_dir else d['ckpt_dir']
        _inst = cls(
            causality=d['causality'],
            ckpt_dir=d['ckpt_dir'],
            qid=d['qid'],
            oid=d['oid'],
            beta=d.get('beta', {}),
            logic=d.get('logic', {}),
        )
        for pid in d['children_pids']:
            _inst.append(Proposition.from_id(d['ckpt_dir'], d['qid'], d['oid'], pid, proxy))
        return _inst


@dataclass
class Proposition:
    sentence: str # usually a one line proposition
    context: str # usually a longer background prompt
    date: str # YYYY-MM-DD, proposition query date, as the real world facts change over time
    ckpt_dir: str 
    proxy: 'Proxy' = None # to provide external context   
    children: 'RelationalPropositions' = None # a list of children propositions that support the proposition
    proof: 'Proof' = None # the proof of the proposition
    parent: 'Proposition' = None # the parent proposition
    pid: str = None # the proposition id, its a variable name A, B, C, ..., root is always P0
    reasoning: List[str] = field(default_factory=list) # the reasoning of the decomposition of the children, only available in top
    oid: str = None # the option id 
    qid: str = None # the query id 
    state: dict = field(default_factory=dict) # the state of the proposition


    def reset(self):
        self.proof = None
        attempt_dir = pjoin(self.ckpt_folder, 'attempts')
        if pexists(attempt_dir):
            n_attempts = len(os.listdir(attempt_dir))
            new_attempt_dir = pjoin(attempt_dir, f'attempt_{n_attempts}')
            # move the entire folder besides the attempts folder
            _files = os.listdir(self.ckpt_folder)
            if 'attempts' in _files:
                _files.remove('attempts')
            if len(_files) == 0:
                return
            mkdirs(new_attempt_dir)
            for _file in _files:
                shutil.move(pjoin(self.ckpt_folder, _file), new_attempt_dir)

    @property
    def prompt(self):
        return f'''The current date is {self.date}. 
Here is the proposition to analyze:
{self.sentence}

Here is the context about this proposition:
{self.context}
''' 
    
    @property
    def session_name(self): # for sandbox
        header = hash_str(f'{self.qid}_{self.oid}', length=10)
        return f'{header}_{self.pid}'

    @property
    def notebook_file(self): 
        _file = pjoin(self.ckpt_folder, 'session', f'{self.session_name}.ipynb')
        return _file if pexists(_file) else None

    @property
    def root(self):
        if self.is_root:
            return self
        else:
            return self.parent.root
        
    @property
    def nodes(self):
        if self.children is None:
            return [self.pid]
        else:
            _nodes = [self.pid]
            for child in self.children.propositions.values():
                _nodes.extend(child.nodes)
            return _nodes

    @property
    def all_nodes(self):
        return self.root.nodes
    
    @property
    def n_nodes(self):
        return len(self.all_nodes)
    
    @property
    def children_nodes(self):
        if self.children is None:
            return []
        else:
            return [child for child in self.children.propositions.values()]
    
    @property
    def children_probabilities(self):
        if self.children is None:
            return {}
        else:
            return {child.pid: child.proof.p_true for child in self.children.propositions.values()}

    def get_node(self, pid: str) -> 'Proposition':
        return self.root._get_node(pid)

    def _get_node(self, pid: str) -> 'Proposition':
        if self.pid == pid:
            return self
        else:
            if self.children is None:
                return None
            else:
                for child in self.children_nodes:
                    return child._get_node(pid)

    def to_dict(self):
        return {
            'pid': self.pid,
            'sentence': self.sentence,
            'context': self.context,
            'date': self.date,
            'ckpt_dir': self.ckpt_dir,
            'proxy': self.proxy.to_dict(),
            'children': self.children.to_dict() if self.children else None,
            'proof': self.proof.to_dict() if self.proof else None,
            'reasoning': self.reasoning,
            'oid': self.oid,
            'qid': self.qid,
        }
    

    @property
    def analyzed(self):
        if self.state.get('analyze_done', False):
            return True
        elif self.children is not None: # NOTE: a PATCH, since it should only be saved when analyze done, but not very safe
            return True
    
    def save_dialog(self, dialog: Dialog, name: str):
        _dialog_dir = pjoin(self.ckpt_folder, 'dialogs')
        mkdirs(_dialog_dir)
        save_json(pjoin(_dialog_dir, f'{name}.json'), dialog.to_dict())

    def save_object(self, obj: dict, name: str):
        _obj_dir = pjoin(self.ckpt_folder, 'objects')
        mkdirs(_obj_dir)
        save_json(pjoin(_obj_dir, f'{name}.json'), obj)
    
    def analyze_done(self, dialog: Dialog):
        self.state['analyze_done'] = True # NOTE: this is a hack to check if the analyze is done
        self.save_dialog(dialog, 'analysis')
        self.save(recursion=True)

    @property
    def ckpt_folder(self):
        return pjoin(self.ckpt_dir, 'ckpts', self.qid, self.oid, self.pid)

    def save_state(self):
        save_json(pjoin(self.ckpt_folder, 'state.json'), self.state)  

    def load_state(self):
        return load_json(pjoin(self.ckpt_folder, 'state.json')) if pexists(pjoin(self.ckpt_folder, 'state.json')) else {}

    def save(self, recursion: bool = False, ext: str = ''): 
        mkdirs(self.ckpt_folder)
        self.save_state()
        save_json(pjoin(self.ckpt_folder, f'proposition{ext}.json'), self.to_dict())  
        if recursion:
            if self.children:  
                for child in self.children.propositions.values():
                    child.save(recursion=True)

    def edit(self, key: str, p_true: float, proof: str = None, key_factor: str = None): # NOTE: use this to implement scenerio analysis with resynthesize, the resynthesize will stop by the edited nodes
        assert 0 <= p_true <= 1, 'New probability must be between 0 and 1'
        self.proof = Proof(p_true=p_true, proof=proof, key_factor=key_factor)
        self.save(ext=f'_{key}')

    @classmethod
    def from_dict(cls, d: dict, proxy: 'Proxy' = None, ckpt_dir: str = None):
        # assert proxy is not None, 'Proxy is required'
        if d.get('proxy', None) and proxy is not None:
            proxy = proxy.reload(**d['proxy'])
        _inst = cls(
            pid=d['pid'],
            sentence=d['sentence'],
            context=d['context'],
            date=d['date'],
            ckpt_dir=ckpt_dir if ckpt_dir else d['ckpt_dir'],
            proxy=proxy,
            children=RelationalPropositions.from_dict(
                d['children'], proxy, ckpt_dir) if d['children'] else None,
            proof=Proof.from_dict(d['proof']) if d['proof'] else None,
            reasoning=d['reasoning'],
            oid=d['oid'],
            qid=d['qid'],
        )
        _inst.load_state()
        if _inst.children:
            for c in _inst.children.propositions.values():
                c.parent = _inst
        return _inst

    @classmethod
    def from_id(cls, ckpt_dir: str, qid: str, oid: str, pid: str, proxy: 'Proxy'):
        _path = pjoin(ckpt_dir, 'ckpts', qid, oid, pid, 'proposition.json')
        if not pexists(_path):
            raise FileNotFoundError(f"Proposition {pid} not found in {_path}")
        return cls.from_dict(load_json(_path), proxy, ckpt_dir) 

    @property
    def json(self):
        return {
            'proposition': self.sentence,
            'context': self.context,
            'date': self.date,
            'children': self.children.json if self.children else None,
        }
    
    def _add_child(self, child: 'Proposition'):
        if self.children is None:
            self.children = RelationalPropositions(
                qid=self.qid,
                oid=self.oid,
                ckpt_dir=self.ckpt_dir,
            )
        self.children.append(child)
        child.parent = self
        
    def set_causality(self, causality: str):
        self.children.causality = causality
    
    def add_reasoning(self, reasoning: str):
        self.reasoning.append(reasoning)

    def set_beta(self, beta: Dict[str, float]):
        self.children.beta = beta

    def set_logic(self, logic: Dict[str, Any]):
        self.children.logic = logic

    @property
    def reasoning_str(self):
        return f'\n{"-"*100}\n'.join(self.reasoning)

    def add_children(self, nodes: Dict[str, Any], root: str):
        if self.pid == root:
            if root in nodes: # otherwise, its a leaf node
                root_node = nodes[root]
                for child_id, child_sentence in root_node['children'].items():
                    child = Proposition(
                        sentence=child_sentence, 
                        context=f'Parent proposition: {self.sentence}', 
                        proxy=self.proxy,
                        date=self.date,
                        ckpt_dir=self.ckpt_dir,
                        pid=child_id,
                        oid=self.oid,
                        qid=self.qid,
                    )
                    self._add_child(child)
                    child.add_children(nodes, child_id)
                self.set_causality(root_node['causality'])
        else:
            for child in self.children.propositions.values():
                child.add_children(nodes, root)
    
    def prove(self, proof: 'Proof'):
        self.proof = proof

    @property
    def n_leaves(self):
        return len(self.leaves) if self.is_root else self.parent.n_leaves

    @property
    def is_leaf(self):
        return self.children is None or len(self.children.propositions) == 0
    
    @property
    def is_root(self):
        return self.parent is None

    @property
    def tree_str(self):
        '''
        example:
        P1
           - P2
              - P3
              - P4
           - P5
              - P6
              - P7
        '''
        return self.get_tree_str(self)
    
    def get_tree_str(self, p: 'Proposition', level: int = 1, max_len = 1000) -> str:
        decs = []
        if p.proved:
            decs.append('proved')
        if p.is_leaf:
            decs.append('leaf')
            _sentence = p.sentence[:max_len] + '...' if len(p.sentence) > max_len else p.sentence
            return f' - {p.pid}: {_sentence} ({", ".join(decs)})\n'
        else:
            decs = f' ({", ".join(decs)})' if decs else ''
            _sentence = p.sentence[:max_len] + '...' if len(p.sentence) > max_len else p.sentence
            _str = f' - {p.pid}: {_sentence}{decs}\n'
            if p.children is not None:
                for c in p.children.propositions.values():
                    _str += '   ' * level + self.get_tree_str(c, level+1, max_len)
            return _str
    
    @property
    def leaves(self):
        if self.is_leaf:
            return [self]
        else:
            _leaves = []
            if self.children is not None:
                for c in self.children.propositions.values():
                    _leaves.extend(c.leaves)
            return _leaves

    def resynthesized(self, key: str):
        ext = f'_{key}' if key else ''
        save_path = pjoin(self.ckpt_folder, f'proposition{ext}.json')
        if pexists(save_path):
            return True
        else:
            return False

    def load_key(self, resynthesize_key: str):
        ext = f'_{resynthesize_key}' if resynthesize_key else ''
        save_path = pjoin(self.ckpt_folder, f'proposition{ext}.json')
        if pexists(save_path):
            return Proposition.from_dict(load_json(save_path), proxy=self.proxy, ckpt_dir=self.ckpt_dir)
        else:
            raise FileNotFoundError(f"Resynthesized proposition {self.pid} not found in {save_path}")


    @property
    def proved(self):
        return self.proof is not None
    
    @property
    def p_true(self):
        return self.proof.p_true if self.proved else None
    



@dataclass
class Proof:
    proof: str
    p_true: float
    key_factor: str = None

    def to_dict(self):  
        return {
            'proof': self.proof,
            'p_true': self.p_true,
            'key_factor': self.key_factor,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            proof=d['proof'],
            p_true=d['p_true'],   
            key_factor=d['key_factor'],
        )


@dataclass
class Option:
    index: int # index of the option within the task
    history: List[dict]
    horizon: int # in days, a desired time horizon
    date: dt.datetime # timestamp of taking the option
    title: str
    description: str
    pair_idx: int # index of the pairs (long, short)
    category: str # category of the option
    is_short: bool = False # same index with the long option
    value: float = None # end value if invest 1, assume infinite leverage, used when deployment
    sr: float = None
    last_date: dt.datetime = None # timestamp of last available data, may not available in deployment


    @property
    def npv(self, discount_rate: float = 0.05):
        assert self.value is not None, "Value is not set"
        if self.horizon is None:
            return self.value
        return self.value / (1+discount_rate)**(self.horizon/252)
    
    @property
    def duration(self):
        return self.date, self.date + dt.timedelta(days=self.horizon)
    
    @property
    def duration_str(self):
        begin, end = self.duration
        return f"{begin.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}"
    
    @property
    def overview(self):
        ls_str = 'Long' if not self.is_short else 'Short'
        _overview = f"Option {self.pair_idx} ({ls_str}): {self.title} ({self.duration_str})"
        if self.value is not None:
            _overview += f", NPV: {self.npv:.2f}"
        if self.sr is not None:
            _overview += f", SR: {self.sr:.2f}"
        return _overview

    def set_eval(self, value: float, sr: float = None, last_date: dt.datetime = None):
        self.value = value
        self.sr = sr
        self.last_date = last_date
    
    @property
    def evaluable(self):
        return self.value is not None
    
    def to_dict(self, save_history: bool = False):
        if save_history:
            history = [h['datetime'].isoformat() for h in self.history]
        else:
            history = None
        return {
            'index': self.index,
            'history': history,
            'horizon': self.horizon,
            'date': self.date.isoformat(),
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'pair_idx': self.pair_idx,
            'is_short': self.is_short,
            'value': self.value,
            'sr': self.sr,
            'last_date': self.last_date.isoformat() if self.last_date else None
        }

    @classmethod
    def from_dict(cls, d: dict):
        if d['history'] is not None:
            history = [dt.datetime.fromisoformat(h) for h in d['history']]
        else:
            history = None
        return cls(
            index=d['index'],
            history=history,
            horizon=d['horizon'],
            date=dt.datetime.fromisoformat(d['date']),
            title=d['title'],
            description=d['description'],
            category=d['category'],
            pair_idx=d['pair_idx'],
            is_short=d['is_short'],
            value=d['value'],
            sr=d['sr'],
            last_date=dt.datetime.fromisoformat(d['last_date']) if d['last_date'] else None
        )
    
    @property
    def short_description(self):
        return f"{self.index+1}. {self.title}"


@dataclass
class Task:
    title: str
    category: str
    description: str
    datetime: dt.datetime
    options: List[Option]
    long_only: bool = False
    one_side_only: bool = False
    # points: List[float] # normalized points

    def config(self, long_only: bool = False, one_side_only: bool = False):
        self.long_only = long_only
        self.one_side_only = one_side_only

    @property
    def prompt(self):
        if self.category in FINANCIAL_CATEGORIES + ['financial']:
            prompt = f"{self.description}"
        elif self.category in PREDMARKET_CATEGORIES:
            prompt = f"{self.title}"
        else:
            prompt = f"{self.title}: {self.description}"
            # raise ValueError(f"Invalid category: {self.category}")
        # prompt += f" The current date is {self.date}.\n"
        prompt += f"\n\nThere are {len(self.options)} options:\n\n"
        for option in self.options:
            prompt += f"{option.short_description}\n\n"
        return prompt
    
    def get_options(self) -> List[Option]:
        if self.long_only or self.one_side_only:
            pairs = self.get_pairs()
            return [pairs[i]['long'] for i in pairs if pairs[i]['long'] is not None]
        else:
            return self.options
    
    def get_pairs(self) -> Dict[int, Dict[str, Option]]:
        _pairs = {}
        for option in self.options:
            pair_idx = option.pair_idx
            if pair_idx not in _pairs:
                _pairs[pair_idx] = {'long': None, 'short': None}
            if option.is_short:
                _pairs[pair_idx]['short'] = option
            else:
                _pairs[pair_idx]['long'] = option
        if self.one_side_only and len(_pairs) == 2 and len(self.options) == 2:
            _long_idx = random.randint(0, 1)
            _pairs = {0: {'long': self.options[_long_idx], 'short': self.options[1-_long_idx]}}
        return _pairs

    @property
    def date(self):
        return self.datetime.strftime('%Y-%m-%d')

    @property
    def overview(self):
        print(f"{self.title} ({self.category})")
        print(self.description)
        print(f"\n{len(self.options)} options:")
        for option in self.options:
            print(option.overview)

    def to_proposition(self, option_idx: int, ckpt_dir: str = None) -> Proposition:
        option = self.options[option_idx]
        _proposition = f"{option.title} is the best option"
        _background = self.prompt
        return Proposition(sentence=_proposition, context=_background, date=self.date, ckpt_dir=ckpt_dir)

    @property
    def evaluable(self):
        return all(option.evaluable for option in self.options)

    def find_counter(self, option_idx: int) -> Option:
        pairs = self.get_pairs()
        for pair in pairs.values():
            long, short = pair['long'], pair['short']
            if long is not None and short is not None:
                if long.index == option_idx:
                    return short
                elif short.index == option_idx:
                    return long
        return None

    def eval(self, option_idx: int):
        if option_idx >= len(self.options):
            raise ValueError(f"Option index {option_idx} is out of range")
        option = self.options[option_idx]
        assert option.evaluable, "Option is not evaluable"  
        return option.npv
    
    @property
    def npvs(self):
        return {idx: option.npv for idx, option in enumerate(self.options)}
    
    def npv(self, option_idx: int):
        return self.options[option_idx].npv
    
    @property
    def npv_stats(self):
        return {
            'min': self.worst_npv,
            'max': self.best_npv,
            'mean': self.mean_npv,
            'median': self.median_npv
        }
    
    @property
    def answer(self):
        if self.evaluable:
            return max(range(len(self.options)), key=lambda x: self.options[x].npv)
        else:
            return None

    @property
    def best_npv(self):
        return max(self.npvs.values())

    @property
    def worst_npv(self):
        return min(self.npvs.values())

    @property
    def mean_npv(self):
        return sum(self.npvs.values()) / len(self.npvs)

    @property
    def median_npv(self):
        return sorted(self.npvs.values())[len(self.npvs) // 2]

    def to_dict(self):
        return {
            'title': self.title,
            'category': self.category,
            'description': self.description,
            'datetime': self.datetime.isoformat(),
            'options': [option.to_dict() for option in self.options]
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            title=d['title'],
            category=d['category'],
            description=d['description'],
            datetime=dt.datetime.fromisoformat(d['datetime']),
            options=[Option.from_dict(option) for option in d['options']]
        )
    


