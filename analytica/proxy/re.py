from analytica.proxy.modules import PROXY_REGISTRY
from analytica.proxy.modules.pm_proxy import get_polymarket_eod, get_kalshi_candles_eod
from analytica.const import Task, Option, FINANCIAL_CATEGORIES, PREDMARKET_CATEGORIES, BASIC_PROXIES, BASE_COMMISSION, RETURN_BOUND, Proposition
import inspect
import argparse
import requests
import analytica.utils as U
import lllm.utils as LU
import os
import json
from typing import List
import datetime as dt
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np
import pandas as pd
from itertools import combinations
from lllm.proxies import BaseProxy
from datasets import load_dataset



FILE_PATH = os.path.dirname(os.path.abspath(__file__)) # /home/junyanc/analytica/analytica/proxy
ETC_PATH = U.pjoin(FILE_PATH,'etc')

_kf_data = U.load_csv(U.pjoin(ETC_PATH,'daily-treasury-rates.csv'))
_kf_data.set_index('Date', inplace=True)
_kf_data.index = pd.to_datetime(_kf_data.index, format='%m/%d/%y')
_kf_data.sort_index(inplace=True)
_colname = '4 WEEKS BANK DISCOUNT'
df = _kf_data[[_colname]]
df[['RF']] = df[[_colname]]/252
RF_FIN = df[['RF']]



class Proxy:
    """
    Runtime Environment
    Note: best using it with the maker class, as it will automatically set the cutoff date.
    In deployment, the cutoff date is not needed.

    This class is used to proxy API requests to the appropriate API.
    cutoff_date: the date to cutoff the data, if None, then use the latest date
    activate_proxies: the list of proxies to activate, if None, then use the default proxies
    """
    def __init__(self, 
                 activate_proxies: List[str] = None, 
                 cutoff_date: str = None, 
                 deploy_mode: bool = False):
        self.proxies: dict[str, BaseProxy] = {}
        self.registry: dict[str, dict] = {}
        self.deploy_mode = deploy_mode
        self.cutoff_date = cutoff_date if not deploy_mode else None
        if activate_proxies is None:
            activate_proxies = BASIC_PROXIES
        # U.cprint(f"Activating proxies: {activate_proxies}", color='y')
        self.activate_proxies = activate_proxies
        for proxy_name in activate_proxies:
            proxy = PROXY_REGISTRY[proxy_name]
            try:
                self.proxies[proxy._proxy_path] = proxy(self.cutoff_date_str)
                self.proxies[proxy._proxy_path]._proxy_path = proxy._proxy_path
            except Exception as e:
                U.cprint(f"Error initializing proxy {proxy._proxy_path}: {e}", color='r')
                continue
            self.registry[proxy._proxy_path] = {
                'name': proxy._proxy_name,
                'description': proxy._proxy_description,
                'doc_string': inspect.getdoc(proxy),
            }

    def reload(self, activate_proxies: List[str] = None, cutoff_date: str = None, deploy_mode: bool = False):
        self.__init__(activate_proxies, cutoff_date, deploy_mode)
        return self

    def to_dict(self):
        return {
            'activate_proxies': self.activate_proxies,
            'cutoff_date': self.cutoff_date_str,
            'deploy_mode': self.deploy_mode,
        }

    @classmethod
    def from_dict(cls, d: dict): return cls(**d)

    @property
    def cutoff_date_str(self):
        if self.deploy_mode:
            return None
        elif self.cutoff_date is None:
            return None
        else:
            return self.cutoff_date.strftime('%Y-%m-%d')

    @property
    def cutoff_date(self):
        if self.deploy_mode:
            return None
        else:
            return self._cutoff_date

    @cutoff_date.setter
    def cutoff_date(self, cutoff_date: str = None):
        if self.deploy_mode:
            cutoff_date = None
        if cutoff_date is None:
            self._cutoff_date = None
        else:
            self._cutoff_date = dt.datetime.strptime(cutoff_date, '%Y-%m-%d')
        for _proxy in self.proxies:
            self.proxies[_proxy].cutoff_date = cutoff_date

    def deploy(self):
        self.deploy_mode = True
        self.cutoff_date = None

    def develop(self):
        self.deploy_mode = False

    def parse_path(self, full_path: str):
        try:
            proxy_path, endpoint = full_path.split('/', maxsplit=1)
            return proxy_path, endpoint
        except ValueError as e:
            raise ValueError(f"Failed to parse path: {full_path}, {e}")

    def __call__(self, full_path: str, params: dict) -> dict:
        proxy_path, endpoint = self.parse_path(full_path)
        return self.proxies[proxy_path](endpoint, params)
    
    def prompt_proxy(self): # prompt the agent to choose the best proxy and api
        raise NotImplementedError("This method should be implemented by the subclass")

    def auto_test(self, proxy_path: str = None, skip_k: int = 0):
        test_proxies = proxy_path.split(',') if proxy_path else self.proxies.keys()
        for proxy in test_proxies:
            self.proxies[proxy].auto_test(skip_k)

    def _prompt_api(self, proxy_path: str, indent = '', additional_doc: bool = True):
        _data = self.registry[proxy_path]
        _proxy = self.proxies[proxy_path]
        _prompt = f''
        for key in _data:
            if key != 'doc_string':
                _prompt += f'{indent} - {key}: {_data[key]}\n'
            else:
                _space = indent*2+'   '
                lines = [_space+i for i in _data[key].split('\n')]
                _prompt += f'{indent} - doc string:\n{_space}---\n{'\n'.join(lines)}\n{_space}---\n' 
                if additional_doc:
                    _additional_doc = _proxy.additional_doc(indent=_space+'  ')
                    _prompt += f'{indent} - additional doc:\n{_additional_doc}\n'
        _prompt += '\n'
        return _prompt

    @property
    def api_catalog(self):
        _prompt = 'API catalog:\n'
        for proxy in self.registry:
            _prompt += f' - {proxy}\n'
            _prompt += self._prompt_api(proxy, indent='  ')
        return _prompt
    
    @property
    def call_directory(self, by_cat: bool = False):
        _prompt = 'Endpoint directory for each API:\n'
        for _api_path, _proxy in self.proxies.items():
            _prompt += f' - {_api_path}\n'
            _prompt += _proxy.endpoint_directory(indent='  ', by_cat=by_cat)
        return _prompt

    @property
    def api_directory(self):
        _prompt = 'API directory:\n'
        indent = '  '
        for proxy in self.registry:
            _prompt += f' - {proxy}\n'
            _prompt += self._prompt_api(proxy, indent=indent)
            _prompt += f'{indent} - endpoints:\n'
            _proxy = self.proxies[proxy]
            _prompt += _proxy.endpoint_directory(indent=indent*2)
        return _prompt
    
    def _api_prompt(self,full_path: str, indent: str = ''):
        proxy_path, endpoint = self.parse_path(full_path)
        _proxy = self.proxies[proxy_path]
        _data = _proxy.registry[endpoint]
        _prompt = f'{indent} - {full_path}\n'
        if _data["name"]:
            _prompt += f'{indent}    - name: {_data["name"]}\n'
        _prompt += f'{indent}    - category: {_data["category"]}\n'
        if _data["sub_category"]:
            _prompt += f'{indent}    - sub_category: {_data["sub_category"]}\n'
        _prompt += f'{indent}    - description: {_data["description"]}\n'
        if _data["doc_string"].strip():
            _indent = '      '+indent
            lines = [_indent+i for i in _data["doc_string"].strip().split('\n')]
            _prompt += f'{indent}    - doc_string:\n{_indent}---\n{'\n'.join(lines)}\n{_indent}---\n'
        _params = _data["params"]
        _prompt += f'{indent}    - params:\n'
        for param in _params:
            _indent = '        '+indent
            _type, _example = _params[param]
            _type = _type.__name__ if _type!='date' else 'str (date)'
            _required = param.endswith('*')
            param = param.replace('$', '').replace('*', '').replace('#', '')
            _required = ' (required)' if _required else ''  
            _prompt += f'{_indent}"{param}": type: {_type}, example: {_example}{_required}\n'
        _prompt += f'{indent}    - example response: {_data["response"]}\n'
        return _prompt

    def retrieve_api_docs(self,full_paths: str | List[str], additional_doc: bool = False): # if include additional doc in directory already, then set additional_doc to False, by default its in directory already
        paths = {}
        if isinstance(full_paths, str):
            full_paths = [full_paths]
        self.check_paths(full_paths)
        for full_path in full_paths:
            proxy_path, _ = self.parse_path(full_path)
            if proxy_path not in paths:
                paths[proxy_path] = []
            paths[proxy_path].append(full_path)
        _prompt = 'Endpoint details:\n'
        for proxy_path in paths:
            _proxy = self.proxies[proxy_path]
            _prompt += f' - {proxy_path}:\n'
            for full_path in paths[proxy_path]:
                _prompt += self._api_prompt(full_path, indent='  ')
            _prompt += '\n'
            if additional_doc and len(_proxy.additional_docs) > 0:
                _prompt += f'   * Additional documentation for {proxy_path} APIs:\n'
                _prompt += _proxy.additional_doc(indent='     ')
            _prompt += '\n'
        return _prompt

    def check_paths(self, paths: List[str]):
        if isinstance(paths, str):
            paths = [paths]
        errors = []
        for path in paths:
            proxy_path, endpoint = self.parse_path(path)
            if proxy_path not in self.proxies:
                errors.append(f"Invalid path: {path}, {proxy_path} not found in the proxy registry")
            if endpoint not in self.proxies[proxy_path].registry:
                errors.append(f"Invalid path: {path}, {endpoint} not found in the registry of {proxy_path}")
        if errors:
            raise ValueError(f"Invalid paths:\n{'\n'.join(errors)}\nPlease check the paths and try again. Remember to use the *full path* of the API from the *API directory*.")
        return True






def eod_to_df(eod: List[dict], to_date: bool = True):
    df = pd.DataFrame(eod)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    if to_date:
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
    return df


@dataclass
class Market:
    ''' class representing a market, expose the data before begin to user
    
    An event has multiple markets, each market corresponds to an option.
    We may use the option to index the market.
    '''
    title: str
    begin: dt.datetime
    end: dt.datetime
    eod: List[dict] # [{"datetime": dt.datetime, "close": float}]
    volume: float
    metadata: dict
    description: str
    category: str

    def slice(self, begin: dt.datetime, end: dt.datetime = None):
        history = [d for d in self.eod if d['datetime'] <= begin]
        future = [d for d in self.eod if begin <= d['datetime'] <= end]
        return history, future

    def build_data(self,cutoff_date: dt.datetime, time_horizon: int = None): # sharp ratio
        time_horizon = U.get_days(time_horizon)
        end = cutoff_date + dt.timedelta(days=time_horizon) if time_horizon else self.end
        history, future = self.slice(cutoff_date, end)
        end_date = max([d['datetime'] for d in future])
        df = eod_to_df(future)
        df['daily_return'] = df['close'].pct_change()
        df.dropna(inplace=True)
        df_rf = pd.merge(df, RF_FIN, left_index=True, right_index=True)
        adjusted_return = (df_rf['daily_return'] - df_rf['RF'])
        sr_long = float(np.mean(adjusted_return)/np.std(adjusted_return)*np.sqrt(252))
        sr_short = float(np.mean(-adjusted_return)/np.std(-adjusted_return)*np.sqrt(252))
        gain = future[-1]['close']/(future[0]['close']+BASE_COMMISSION)-1
        gain = min(gain, RETURN_BOUND)
        value_long = gain+1
        value_short = 1-gain # suppose infinite leverage
        span_days = (end - cutoff_date).days
        return history, value_long, value_short, sr_long, sr_short, span_days, end_date


@dataclass
class Event:
    ''' class representing a events, expose the data before begin to user
    '''
    title: str
    tags: List[str]
    markets: List[Market]
    volume: float
    metadata: dict
    description: str = None
    category: str = None
    
    @property
    def min_span(self):
        _begin = max([market.begin for market in self.markets])
        _end = min([market.end for market in self.markets])
        return _begin, _end
    
    def to_eval(self, cutoff_date: dt.datetime, time_horizons: List[int], allow_short: bool):
        options = []
        index = 0
        for market in self.markets:
            if self.category in FINANCIAL_CATEGORIES + ['financial']:
                if not isinstance(time_horizons, list):
                    time_horizons = [time_horizons]
                long_str = f'Long {market.category} {market.title} and hold for'
                short_str = f'Short {market.category} {market.title} and hold for'
            elif self.category in PREDMARKET_CATEGORIES:
                time_horizons = [None]
                _title = market.title.replace('(Yes?)','').replace('(?)','')
                long_str = f'Bet on: "{_title}"'
                short_str = f'Bet against: "{_title}"'
            else:
                raise ValueError(f"Invalid category: {self.category}")
            for time_horizon in time_horizons:
                if self.category in FINANCIAL_CATEGORIES + ['financial']+ PREDMARKET_CATEGORIES:
                    history, value_long, value_short, sr_long, sr_short, span_days, end_date = market.build_data(cutoff_date, time_horizon)
                else:
                    raise ValueError(f"Invalid category: {self.category}")
                if self.category in FINANCIAL_CATEGORIES + ['financial']:
                    _long_str = f'{long_str} {U.to_str_span(span_days)}'
                    _short_str = f'{short_str} {U.to_str_span(span_days)}'
                else:
                    _long_str = f'{long_str}'
                    _short_str = f'{short_str}'
                long_option = Option(len(options), history, span_days, cutoff_date, _long_str, market.description, index, 
                                     is_short=False, value=value_long, sr=sr_long, last_date=end_date, category=market.category)
                options.append(long_option)
                if allow_short:
                    short_option = Option(len(options), history, span_days, cutoff_date, _short_str, market.description, index, 
                                          is_short=True, value=value_short, sr=sr_short, last_date=end_date, category=market.category)
                    options.append(short_option)
                index += 1
        return Task(self.title, self.category, self.description, cutoff_date, options)



@dataclass
class Query:
    task: Task
    proxy: Proxy
    ckpt_dir: str
    qid: str

    def __post_init__(self):
        if self.qid is None:
            _qid = U.replace_special_chars(self.task.title).lower()[:10]
            self.qid = f'{_qid}_{U.dt_now_str()}_{U.random_str(6)}'
        else:
            self.load_state()

    @property
    def options(self):
        return self.task.options
    
    @property
    def prompt(self):
        return self.task.prompt

    def to_proposition(self, option_idx: int) -> Proposition:
        proposition = self.task.to_proposition(option_idx, self.ckpt_dir)
        proposition.proxy = self.proxy
        return proposition
    
        
    @property
    def ckpt_folder(self):
        return U.pjoin(self.ckpt_dir, 'ckpts', self.qid)

    def load_state(self):
        assert self.ckpt_dir is not None, "CKPT path is not set"
        _path = U.pjoin(self.ckpt_folder, 'state.json')
        _state = U.load_json(_path) if U.pexists(_path) else {}
        _state['option_state'] = {int(k): v for k, v in _state.get('option_state', {}).items()}
        return _state
    
    def save_state(self, state: dict):
        assert self.ckpt_dir is not None, "CKPT path is not set"
        U.save_json(U.pjoin(self.ckpt_folder, 'state.json'), state)

    def init_analysis(self, option_idx: int) -> Proposition: # save the session id of the top propositions
        U.mkdirs(self.ckpt_folder)
        _state = self.load_state()
        _state['option_state'] = _state.get('option_state', {})
        
        _path = None
        oid = None
        if option_idx in _state['option_state']:
            oid = _state['option_state'][option_idx]['oid']
            _path = U.pjoin(self.ckpt_folder, oid, 'P0', 'proposition.json')
        if _path and U.pexists(_path):
            U.cprint(f'Loading proposition from {_path}', color='y')
            proposition = Proposition.from_dict(U.load_json(_path), self.proxy, self.ckpt_dir)
        else:
            proposition = self.to_proposition(option_idx)
            if oid is None:
                _oid = U.replace_special_chars(self.task.options[option_idx].title).lower().replace('_', '+')[:10]
                proposition.oid = f'{_oid}_{U.dt_now_str()}_{U.random_str(6)}'
                _state['option_state'][option_idx] = {'oid': proposition.oid}
            else:
                proposition.oid = oid
            proposition.pid = 'P0'
        proposition.qid = self.qid
        self.save_state(_state)
        return proposition
    


class Maker:
    '''
    Given a ticker, begin, and end date, generate the market, and the evaluation.
    '''
    def __init__(self, config: dict, deploy_mode: bool = False, ckpt_dir: str = None):
        self.config = config
        self.deploy_mode = deploy_mode
        self.kalshi_events = U.load_json(U.pjoin(ETC_PATH,'kalshi_events.json'))
        self.polymarket_events = U.load_json(U.pjoin(ETC_PATH,'polymarket_events.json'))
        self.cleaned_tickers = U.load_json(U.pjoin(ETC_PATH,'cleaned_tickers.json')) # clearing buggy, problematic tickers
        self.kalshi_events = {e['event_ticker']:e for e in self.kalshi_events}
        self.polymarket_events = {e['ticker']:e for e in self.polymarket_events}
        self._preload_events = {}
        self._proxy = Proxy(self.config['activate_proxies'], deploy_mode=deploy_mode)
        self.ckpt_dir = ckpt_dir
        self.cross_selection_total = config['cross_selection_total']
        self.cross_selection_max = config['cross_selection_max']
        
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        self.kalshi_api_key = os.getenv("KALSHI_API_KEY_ID")
        self.knowledge_cutoff = U.dts_to_dt(str(self.config['knowledge_cutoff']))

        if self.config['activate_events'] is None:
            self.config['activate_events'] = FINANCIAL_CATEGORIES + PREDMARKET_CATEGORIES
        
        self.load_tickers(self.config['activate_events'])

    def load_tickers(self,activate_events: List[str]):
        U.cprint(f"Activating events: {activate_events}", color='y')

        _tickers = {
            'stock': list(set(self.cleaned_tickers['stock'])),
            'crypto': list(set(self.cleaned_tickers['crypto'])),
            'forex': list(self.cleaned_tickers['forex']),
            'commodity': list(set(self.cleaned_tickers['commodity'])),
            'index': list(set(self.cleaned_tickers['index'])),
            'fund': list(set(self.cleaned_tickers['fund'])),
            'kalshi': list(set(self.cleaned_tickers['kalshi'])),
            'polymarket': list(set(self.cleaned_tickers['polymarket']))
        }
        self.tickers = {cat: _tickers[cat] for cat in activate_events}
        filtered = self.filter_events()
        U.cprint(f"{filtered} forcasting events filtered out.", color='y')
        self.xtickers = self.build_cross_selection()
        filtered = self.filter_events(cross_selection=True)
        U.cprint(f"{filtered} cross-selection events filtered out.", color='y')
        # self.overview()

        random.seed(self.config['random_seed'])
        for cat in self.tickers:
            random.shuffle(self.tickers[cat])
        self.ticker_sequence = [t for cat in self.tickers for t in self.tickers[cat]]
        self.xticker_sequence = [t for cat in self.xtickers for t in self.xtickers[cat]]
        random.shuffle(self.ticker_sequence)

    def deploy(self):
        self.deploy_mode = True
        self.cutoff_date = None
        self._proxy.deploy()

    def develop(self):
        self.deploy_mode = False
        self._proxy.develop()

    def overview(self):
        # U.cprint(f"Overview of the maker:", color='y')
        total_tickers = sum([len(self.tickers.get(category,[])) for category in self.tickers])
        total_xtickers = sum([len(self.xtickers[category]) for category in self.xtickers])
        total_tasks = total_tickers + total_xtickers
        fin_total = sum([len(self.tickers.get(category,[])) for category in FINANCIAL_CATEGORIES])
        pred_total = sum([len(self.tickers.get(category,[])) for category in PREDMARKET_CATEGORIES])
        U.cprint(f"Forcasting tasks: {total_tickers}", color='y')
        U.cprint(f"  - Financial Markets: {fin_total} ({fin_total/total_tickers*100:.2f}%)", color='y')
        for category in FINANCIAL_CATEGORIES:
            _tickers = self.tickers.get(category,[])
            U.cprint(f"    - {category}: {len(_tickers)} ({len(_tickers)/total_tickers*100:.2f}%)", color='y')
        U.cprint(f"  - Predictive Markets: {pred_total} ({pred_total/total_tickers*100:.2f}%)", color='y')
        for category in PREDMARKET_CATEGORIES:
            _tickers = self.tickers.get(category,[])
            U.cprint(f"    - {category}: {len(_tickers)} ({len(_tickers)/total_tickers*100:.2f}%)", color='y')
        U.cprint(f"Cross-selection tasks: {total_xtickers}", color='y')
        for category in self.xtickers:
            U.cprint(f"  - {category}: {len(self.xtickers[category])} ({len(self.xtickers[category])/total_xtickers*100:.2f}%)", color='y')
        U.cprint(f"Total tasks: {total_tasks}", color='y')
        print()

    def task_stats(self):
        n_options = []
        npvs = []
        answer_by_category = {}
        for cat in self.tickers:
            answer_by_category[cat] = []
            for ticker in self.tickers[cat]:
                query = self.make_query(ticker)
                n_options.append(len(query.task.options))
                npvs.append([o.npv for o in query.task.options])
                answer_by_category[cat].append(query.task.answer)

        min_npvs = ('Min',[])
        max_npvs = ('Max',[])
        mean_npvs = ('Mean',[])
        median_npvs = ('Median',[])
        std_npvs = ('Std',[])
        for _npv in npvs:
            min_npvs[1].append(min(_npv))
            max_npvs[1].append(max(_npv))
            mean_npvs[1].append(np.mean(_npv))
            median_npvs[1].append(np.median(_npv))
            std_npvs[1].append(np.std(_npv))
        all_npvs = ('All',[])
        for _npv in npvs:
            all_npvs[1].extend(_npv)

        for title, _set in [min_npvs,max_npvs,mean_npvs,median_npvs,std_npvs,all_npvs]:
            print(f'{title}: min: {min(_set):.2f}, max: {max(_set):.2f}, mean: {np.mean(_set):.2f}, median: {np.median(_set):.2f}, std: {np.std(_set):.2f}')
        
        print(f'\nAnswer distribution by category:')
        for cat in answer_by_category:
            print(f'{cat}: {U.list2freq(answer_by_category[cat])}')
        return n_options, npvs

    def filter_events(self, cross_selection: bool = False):
        error_events = U.load_state('error_events')
        missing_data = error_events.get('missing_data', [])
        invalid_span = error_events.get('invalid_span', [])
        error_tickers = missing_data + invalid_span
        _tickers = self.xtickers if cross_selection else self.tickers
        cat_tickers = [(cat,t) for cat in _tickers for t in _tickers[cat]]
        knowledge_cutoff = U.dts_to_dt(str(self.config['knowledge_cutoff']))
        
        _filtered = 0
        _bar = tqdm(cat_tickers, desc="Loading events", total=len(cat_tickers))
        for category, ticker in _bar:
            _bar.set_description(f"Loading: {ticker} ({category})")
            if ticker in error_tickers:
                _tickers[category].remove(ticker)
                _filtered += 1
                continue
            try:
                event = self.make_event(ticker)
                query = self.make_query(ticker)
                if query.task.best_npv == query.task.worst_npv:
                    raise ValueError(f"Best and worst NPV are the same for {ticker}")
            except Exception as e:
                # U.cprint(f"Error making event for {ticker}: {e}", color='r')
                _tickers[category].remove(ticker)
                _filtered += 1
                continue
            _begin, _end = event.min_span
            _begin = max(_begin, knowledge_cutoff)
            if _begin > _end:
                _tickers[category].remove(ticker)
                _filtered += 1
                continue
            min_span = _end - _begin
            if category in FINANCIAL_CATEGORIES:
                min_days = self.config['min_days_financial']
            else:
                min_days = self.config['min_days_predmarket']
            n_markets = len(event.markets)

            if (min_span < dt.timedelta(days=min_days) 
                or n_markets > self.config['max_markets']):
                _tickers[category].remove(ticker)
                _filtered += 1
        if cross_selection:
            self.xtickers = _tickers
        else:
            self.tickers = _tickers
        return _filtered
    
    def find_category(self, ticker: str):
        # for category in FINANCIAL_CATEGORIES + PREDMARKET_CATEGORIES:
        for category in self.tickers:
            if ticker in self.tickers.get(category,[]):
                return category
        raise ValueError(f"Invalid ticker: {ticker}")
    
    def check_event(self, event: Event):
        for market in event.markets:
            assert len(market.eod) > 0, f"No data found for market: {market.title}"
        _begin, _end = event.min_span
        assert _begin <= _end, f"Invalid event, span is invalid: {event.title}"
        if event.category in FINANCIAL_CATEGORIES:
            assert len(event.markets) == 1, f"Financial event {event.title} has more than one market"
        else:
            assert len(event.markets) > 0, f"Predictive event {event.title} has no market"

    def hash_xticker(self, xticker: List[str]):
        return '_vs_'.join(xticker)

    def make_event(self, ticker: str | List[str]): 
        if isinstance(ticker, list) and len(ticker) == 1:
            ticker = ticker[0]
        if isinstance(ticker, list):
            ticker_hash = self.hash_xticker(ticker)
        else:
            ticker_hash = ticker
        if ticker_hash in self._preload_events:
            return self._preload_events[ticker_hash]
        if isinstance(ticker, str):
            category = self.find_category(ticker)
            if category in FINANCIAL_CATEGORIES:
                event = self.make_financial_markets(ticker)
            elif category == 'kalshi':
                event = self.make_kalshi_markets(ticker)
            elif category == 'polymarket':
                event = self.make_polymarket_markets(ticker)
            else:
                raise ValueError(f"Invalid category: {category}")
        elif isinstance(ticker, list):
            event = self.make_multi_financial_markets(ticker)
        else:
            raise ValueError(f"Invalid ticker: {ticker}")
        self.check_event(event)
        self._preload_events[ticker_hash] = event
        return event
    
    def make_query(self, 
            ticker: str | List[str], 
            time_horizons: List[int] = None, 
            allow_short: bool = None,
            qid: str = None # for ckpt
        ): # Make the evaluation based on the event to test the LLM agent
        if isinstance(ticker, list):
            if len(ticker) == 1:
                ticker = ticker[0]
            else:
                event = self.make_multi_financial_markets(ticker)
        else:
            event = self.make_event(ticker)
        if time_horizons is None:
            time_horizons = self.config['time_horizons']
        if allow_short is None:
            allow_short = (event.category in FINANCIAL_CATEGORIES) or (len(event.markets) < 2)
        task = event.to_eval(self.knowledge_cutoff, time_horizons, allow_short)
        proxy = self.init_proxy(task)
        return Query(task, proxy, ckpt_dir=self.ckpt_dir, qid=qid)

    def random_query(self, category: str | List[str] = None, cross_selection: bool = False):
        pool = self.tickers if not cross_selection else self.xtickers
        if category is None:
            _non_empty = [c for c in pool if len(pool[c]) > 0]
            category = random.choice(_non_empty)
        if isinstance(category, list):
            category = random.choice(category)
        assert category in pool, f"Inactive categories: {category}"
        ticker = random.choice(pool[category])
        return self.make_query(ticker)

    def init_proxy(self, task: Task):
        if self.deploy_mode:
            self._proxy.cutoff_date = None
        else:
            self._proxy.cutoff_date = task.date
        return self._proxy

    def make_kalshi_markets(self, ticker: str):
        event = self.kalshi_events[ticker]
        series_ticker = event['series_ticker']
        markets = []

        def repeat_price(candlesticks, i): # FIXME: if its 0, then there is no cost??
            price = candlesticks[i]['price']['close']
            if price is None:
                if i == 0:
                    return float(0)
                else:
                    return repeat_price(candlesticks, i-1)
            else:
                return float(price)

        def to_eod(candlesticks: dict):
            eod =[]
            for i in range(len(candlesticks)):
                ts = candlesticks[i]['end_period_ts']
                eod.append({
                    'datetime':dt.datetime.fromtimestamp(ts),
                    'close':repeat_price(candlesticks, i)
                })
            eod = self.sort_eod(eod)
            return eod

        for m in event['markets']:
            market_eod = get_kalshi_candles_eod(series_ticker, m)
            eod = to_eod(market_eod['candlesticks'])
            begin = U.dts_to_dt(m['open_time'])
            end = U.dts_to_dt(m['close_time'])
            m['source'] = 'kalshi'
            market = Market(
                title=f"{m['title']} ({m['subtitle']}?)",
                begin=begin,end=end,eod=eod,
                volume=float(m['volume']),metadata=m,
                description=f"{m['rules_primary']} {m['rules_secondary']}",
                category='kalshi'
            )
            markets.append(market)
        metadata = {i:event[i] for i in event if i not in ['markets']}
        metadata['source'] = 'kalshi'
        return Event(
            title=event['title'],
            tags=[event['category']],
            markets=markets,
            volume=float(event['volume']) if event['volume'] else None,
            metadata=metadata,
            category='kalshi'
        )

    def make_polymarket_markets(self, ticker: str):
        event = self.polymarket_events[ticker]
        markets = []
        for m in event['markets']:
            token_eods = get_polymarket_eod(m)
            options = list(token_eods.keys())
            eod = [{'datetime':dt.datetime.fromtimestamp(d['t']),'close':float(d['p'])} 
                    for d in token_eods[options[0]]['history']]
            if len(eod) == 0:
                raise ValueError(f"No data found for {ticker}")
            eod = self.sort_eod(eod)
            markets.append(Market(
                title=f"{m['question']} ({options[0]}?)",
                begin=U.dts_to_dt(m['startDate']) if 'startDate' in m else eod[0]['datetime'],
                end=U.dts_to_dt(m['endDate']) if 'endDate' in m else eod[-1]['datetime'],
                eod=eod,
                volume=float(m['volume']) if m['volume'] else None,
                description=m['description'],
                metadata=m,
                category='polymarket'
            ))
        metadata = {i:event[i] for i in event if i not in ['markets']}
        metadata['source'] = 'polymarket'
        tags = [t['label'] for t in event['tags']]
        return Event(
            title=event['title'],
            tags=tags,
            markets=markets,
            volume=float(event['volume']) if event['volume'] else None,
            metadata=metadata,
            description=event.get('description', None),
            category='polymarket'
        )


    def get_financial_eod(self, ticker: str, begin: str = None, end: str = None):
        url = f'https://financialmodelingprep.com/stable/historical-price-eod/light'
        params = {
            'apikey': self.fmp_api_key,
            'symbol': ticker,
            'from': begin,
            'to': end
        }
        return LU.call_api(url, params)
    
    def sort_eod(self, eod: list): # sort the eod by datetime old to new
        return sorted(eod, key=lambda x: x['datetime'])

    def make_financial_markets(self, ticker: str):
        eod = self.get_financial_eod(ticker)
        markets = []
        _eod = [{'datetime':U.dts_to_dt(d['date']),'close':float(d['price'])} for d in eod]
        if len(_eod) == 0:
            raise ValueError(f"No data found for {ticker}")
        _eod = self.sort_eod(_eod)
        mean_volume = sum([float(d['volume']) for d in eod])/len(eod)
        category = self.find_category(ticker)
        markets.append(Market(
            title=f"{ticker}",
            begin=_eod[0]['datetime'],
            end=_eod[-1]['datetime'],
            eod=_eod,
            volume=mean_volume,
            description=f"{ticker}",
            metadata={},
            category=category
        ))
        return Event(
            title=f"{ticker}",
            tags=[],
            markets=markets,
            volume=mean_volume,
            metadata={},
            description=f"Invest in this {category}: {ticker}?",
            category=category
        )
    
    # composed event from multiple financial markets
    def make_multi_financial_markets(self, tickers: List[str]):
        description = 'Invest in these financial markets?'
        markets = []
        for ticker in tickers:
            event = self.make_event(ticker)
            category = event.category
            assert category in FINANCIAL_CATEGORIES, f"Only financial categories are supported for multiple financial markets"
            description += f'\n- {ticker} ({category})'
            markets.extend(event.markets)
        mean_volume = sum([float(m.volume) for m in markets])/len(markets)

        return Event(
            title=f"{' vs '.join(tickers)}",
            tags=[],
            markets=markets,
            volume=mean_volume,
            metadata={},
            description=description,
            category='financial'
        )
    
    def build_cross_selection(self):
        random.seed(42)
        tickers = {}
        for category, n_samples in self.cross_selection_total.items():
            if category != 'any':
                if category not in self.tickers:
                    tickers[category] = []
                    continue
                _pool = list(combinations(self.tickers[category], self.cross_selection_max))
                n_samples = min(n_samples, len(_pool))
                tickers[category] = [list(t) for t in random.sample(_pool, n_samples)]
        n_samples_any = self.cross_selection_total['any']
        tickers['any'] = []
        if len(self.tickers) > 1:
            _pool = []
            for c in FINANCIAL_CATEGORIES:
                if c in self.tickers:
                    _pool.extend(self.tickers[c])
            if len(_pool) > 0:
                n_samples_any = min(n_samples_any, len(_pool))
                _pool = list(combinations(_pool, self.cross_selection_max))
                for _ in range(n_samples_any):
                    while True:
                        _comb = random.choice(_pool)
                        if _comb not in tickers['any']:
                            tickers['any'].append(_comb)
                            break
            tickers['any'] = [list(t) for t in tickers['any']]
        return tickers

    def self_check(self,skip_k: int = 0):
        # preload the data and self check the data
        self.overview()
        test = 0
        all_errors = {}
        for category in self.tickers:
            U.cprint(f"Testing {category}...", color='y')
            bar = tqdm(self.tickers[category])
            for ticker in bar:
                test += 1
                bar.set_postfix(test=test)
                if test < skip_k:
                    continue
                try:
                    query = self.make_query(ticker)
                except Exception as e:
                    e = f'{e} [index: {test}]'
                    U.cprint(f"Error making event for {ticker}: {e}", color='r')
                    all_errors[ticker] = e
                    continue
            U.cprint(f"Finished testing {category}.", color='y')  
            print()

        error_events = U.load_state('error_events')
        if 'missing_data' not in error_events:
            error_events['missing_data'] = []
        if 'invalid_span' not in error_events:
            error_events['invalid_span'] = []
        U.cprint("Finished testing all categories.", color='y')
        if len(all_errors) > 0:
            U.cprint(f"Total errors: {len(all_errors)}", color='r')
            for ticker, error in all_errors.items():
                U.cprint(f"{ticker}: {error}", color='r')
                if 'No data found' in error or 'No clobTokenIds found' in error:
                    error_events['missing_data'].append(ticker) 
                if 'span is invalid' in error:
                    error_events['invalid_span'].append(ticker)
        else:
            U.cprint("All tests passed.", color='g')
        U.save_state('error_events', error_events)

    


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_proxy", "-tp", type=str, default=None)
    parser.add_argument("--cutoff_date", "-c", type=str, default=None)
    parser.add_argument("--skip_k", "-s", type=int, default=0)
    parser.add_argument("--test_maker", "-tm", action='store_true')
    parser.add_argument("--config", "-cfg", type=str, default='base')
    args = parser.parse_args()

    config = U.load_config(U.pjoin('configs', f'{args.config}.yaml'))

    if args.test_proxy:
        proxy = Proxy(cutoff_date=args.cutoff_date, activate_proxies=config['activate_proxies'])
        path = args.test_proxy if args.test_proxy != 'all' else None
        proxy.auto_test(path, args.skip_k)

    if args.test_maker:
        maker = Maker(config)
        # maker.self_check(args.skip_k)



