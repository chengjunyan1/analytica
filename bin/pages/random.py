import streamlit as st
import bin.app_utils as AU


def random():
    st.title('Random')

    AU.side_status()

    system = st.session_state.system
    # system.maker.task_stats()

    st.write(list(system.maker.tickers.keys()))


    query = system.maker.random_query(cross_selection=True)
    st.write(query)


    st.write(query.task.prompt)
    

