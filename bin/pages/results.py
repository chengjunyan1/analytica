import streamlit as st
import bin.app_utils as AU

from analytica.system import SystemBase
from analytica.agent.ae import AgentType
from analytica.const import FINANCIAL_CATEGORIES, PREDMARKET_CATEGORIES
        


def results():
    st.title('Benchmark')

    AU.side_status()

    asys: SystemBase = st.session_state.system


    cols = st.columns([1,1,2,1])
    with cols[0]:
        current_agent = asys.agent.agent_type
        _agent_types = [a for a in AgentType]
        _index = _agent_types.index(current_agent)
        agent_type = st.selectbox('Agent Type', _agent_types, index=_index)
        if agent_type != current_agent:
            asys.rebuild(agent_type)
    
    with cols[1]:
        AU.button_sb_spacer()
        with st.expander('Agent Configs'):
            st.write('Agent Type: ', asys.agent_type)
            st.write('Agent Configs: ', asys.agent.agent_configs)

    with cols[2]:
        activate_events = st.multiselect('Activating Events', 
                       FINANCIAL_CATEGORIES + PREDMARKET_CATEGORIES, 
                       default=asys.maker.config['activate_events'])
        need_reload = activate_events != asys.maker.config['activate_events']

    with cols[3]:
        AU.button_sb_spacer()
        if st.button('Reload', disabled=not need_reload):
            asys.maker.config['activate_events'] = activate_events
            asys.maker.load_tickers(activate_events)


    args = AU.agent_settings(asys.agent_type)

    cols = st.columns(5)

    _total_tickers = len(asys.maker.ticker_sequence)

    with cols[0]:
        n_tickers = st.number_input('N Tickers', min_value=1, max_value=_total_tickers, 
                                    value=_total_tickers, step=1, help='Randomly selected number of tickers to evaluate')

    with cols[1]:
        AU.button_sb_spacer()
        run_eval = st.button('Run Eval', use_container_width=True)

    with cols[2]:
        AU.checkbox_sb_spacer()
        show_detail = st.checkbox('Show Detail', value=False)


    if run_eval:
        asys.evaluate(show_detail=show_detail, n_tickers=n_tickers, agent_args=args)