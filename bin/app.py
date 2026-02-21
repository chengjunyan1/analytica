import sys,os
sys.path.append('.')

import time
import pathlib
import streamlit as st
import analytica.utils as U
import bin.app_utils as AU


custom_args = sys.argv[1:]

DEPLOY_MODE = 'deploy' in custom_args or '--deploy' in custom_args or '-d' in custom_args or 'd' in custom_args



current_dir = pathlib.Path(__file__).parent
logo_path = U.pjoin(current_dir,'assets','ae_uc.svg')

logo=AU.svg_to_image(logo_path)
st.set_page_config(page_title="Analytica", layout="wide",page_icon=logo)


import importlib
from streamlit_theme import st_theme

from streamlit_navigation_bar import st_navbar


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import the parent module first
if not DEPLOY_MODE:
    import bin.pages

    # Function to dynamically import and reload modules
    def import_and_reload(module_name):
        full_module_name = f'bin.pages.{module_name}'
        if full_module_name in sys.modules:
            return importlib.reload(sys.modules[full_module_name])
        return importlib.import_module(full_module_name)

    # Import and reload modules
    home = import_and_reload('home').home
    sources = import_and_reload('sources').sources
    analyze = import_and_reload('analyze').analyze
    viewers = import_and_reload('viewers').viewers
    configs = import_and_reload('configs').configs
    results = import_and_reload('results').results
    random = import_and_reload('random').random
else:
    from bin.pages import home
    from bin.pages import sources
    from bin.pages import analyze
    from bin.pages import viewers
    from bin.pages import configs
    from bin.pages import results
    from bin.pages import random


from analytica.system import build_system



# Setup

@st.cache_resource()
def build(cfg_name,exp_name):
    config = U.load_config(
        U.pjoin(PROJECT_ROOT, 'configs', f'{cfg_name}.yaml'),
    )
    system = build_system(config,exp_name,stream=st)
    return system


cfg_name = 'test'
exp_name = 'test'
system = build(cfg_name,exp_name)


st.session_state.is_deploy = DEPLOY_MODE
st.session_state.current_theme = st_theme()
st.session_state.system = system
st.session_state.project_dir = PROJECT_ROOT



project_dir = current_dir.parent

styles = {
    "nav": {
        # "background-color": "royalblue",
        # "justify-content": "left",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        # "color": "white",
        "padding": "14px",
    },
    # "active": {
    #     "background-color": "white",
    #     "color": "var(--text-color)",
    #     "font-weight": "normal",
    #     "padding": "14px",
    # }
}

urls = {"GitHub": "https://github.com/chengjunyan1/analytica"}

pages = {
    'Analyze': analyze,
    'Benchmark': results, 
    'Source': sources,
    'Viewer': viewers,
    'Config': configs,
}



titles=list(pages.keys())

if not DEPLOY_MODE:
    titles.append('Random')
    pages['Random'] = random

titles.append('GitHub')

pg = st_navbar(
    titles,
    logo_path=logo_path,
    styles=styles,
    urls=urls
)
pages['Home'] = home

if pg is None:
    pg = 'Home'
    

pages[pg]()


