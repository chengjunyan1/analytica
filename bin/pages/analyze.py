import streamlit as st
import bin.app_utils as AU

from analytica.system import SystemBase
from analytica.agent.ae import AgentType



def analyze():
    st.title('Analyze')

    AU.side_status()

    asys: SystemBase = st.session_state.system


    cols = st.columns([1,2])
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

    cols = st.columns(5)

    with cols[0]:
        _category = st.selectbox('Category', list(asys.maker.tickers.keys()))
        _cat_tickers = asys.maker.tickers[_category]

    with cols[1]:
        _ticker = st.selectbox('Ticker', _cat_tickers)
        query = asys.maker.make_query(_ticker)

    with cols[2]:
        AU.button_sb_spacer()
        _RUN = st.button('Run', use_container_width=True)

    with cols[3]:
        AU.checkbox_sb_spacer()
        if st.checkbox('Deploy Mode', value=asys.deploy_mode, 
                       help='Deploy mode will disable the cutoff date.'):
            asys.deploy()
        else:
            asys.develop()
    
    
    args = AU.agent_settings(asys.agent_type)
    
    if _RUN:
        report = asys.call(_ticker, **args)
        # st.write(report)


    