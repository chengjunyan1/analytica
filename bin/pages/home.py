import json
import time
import pathlib
import streamlit as st
import sys,os
from datetime import datetime
import asyncio
# import python_weather

import analytica.utils as U
import bin.app_utils as AU

sys.path.append('.')


current_dir = pathlib.Path(__file__).parent


# async def _getweather(city):
#   # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
#   async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
#     weather = await client.get(city)
#     return weather.temperature
  

def tabs():
    st.subheader('Tabs')

    tabs=st.tabs([
       'Analyze',
       'Sources',
       'Results',
       'Viewers',
       'Configs',
    ])

    with tabs[0]:
      st.markdown('''
The Analyze tab is for the analysis of the information sources.
''')

    with tabs[1]:
      st.markdown('''
The Sources tab is for the information sources and tools.
''')

    with tabs[2]:
      st.markdown('''
The Results tab is for running experiments and analyzing the results.
''')

    with tabs[3]:
      st.markdown('''
The Viewers tab is an analysis tool for the system.
''')

    with tabs[4]:
      st.markdown('''
The Configs tab is for the configuration of the analytical engine.
''')



def howtouse():
    st.markdown('# Welcome to Analytica')


    st.markdown('''
## How to use Analytica

Analytica is an analytical engine.


''')
    



def home():
    AU.side_status()
    howtouse()
    tabs()