import cairosvg
from PIL import Image
import io
import numpy as np
import uuid
import pandas as pd
import os,sys
from art import tprint
import streamlit as st
import yfinance as yf

from analytica.agent.ae import AgentType

sys.path.append('..')
import analytica.utils as U
import pytz
from datetime import datetime, timedelta

CLI_TITLE = 'Analytica'



def svg_to_image(svg_path):
    svg_data = open(svg_path, 'r').read()
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    return Image.open(io.BytesIO(png_data))

def get_bin_dir():
    project_dir = st.session_state['project_dir']
    return U.pjoin(project_dir, 'bin')

def get_latest_price(ticker: str):
    price = yf.Ticker(ticker).info['regularMarketPrice']
    movement = yf.Ticker(ticker).info['regularMarketChangePercent']
    return price, movement

def show_market_price(ticker, label, prec = 1, use_metric = False):
    price, movement = get_latest_price(ticker)
    color = 'green' if movement > 0 else 'red'
    price = round(price, prec)
    if use_metric:
        st.metric(label=label, value=price, delta=f"{movement:+.2f}%")
    else:
        _price = f":{color}[{price} ({movement:+.2f}%)]"
        _label = f"{label}"
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"{_label}")
        with col2:
            st.markdown(f"{_price}")


def show_market_today():
    st.markdown(f"#### The Market Today")
    st.caption("US Stock Market")
    show_market_price('^GSPC', 'S&P 500')
    show_market_price('^IXIC', 'Nasdaq')
    show_market_price('^DJI', 'Dow 30')
    # show_market_price('^RUT', 'Russel')
    show_market_price('^VIX', 'VIX', prec = 2)
    st.caption("Commodities")
    show_market_price('GC=F', 'Gold')
    show_market_price('CL=F', 'Oil')
    st.caption("Interest Rates")
    show_market_price('^TYX', 'T-30Y', prec = 3)
    st.caption("Forex")
    show_market_price('EURUSD=X', 'Eur/Usd', prec = 4)
    st.caption("Crypto")
    show_market_price('BTC-USD', 'Bitcoin')


def side_status(show_market = True):
    bin_dir = get_bin_dir()
    with st.sidebar:
        img_path=U.pjoin(bin_dir,'assets','ana.jpg')
        st.image(img_path)

        st.caption(f"Today is :blue[{datetime.now().strftime('%b %d, %Y, %A')}].")

        if show_market:
            show_market_today()


def spacer(height: int = 28):
    st.markdown(f"<div style='width: 1px; height: {height}px'></div>", unsafe_allow_html=True)

def button_sb_spacer(): # to align with selectbox
    spacer(28)

def checkbox_sb_spacer(): # to align with checkbox
    spacer(36)


def agent_settings(agent_type):
    args = {}
    with st.expander('Additional Configs', expanded=True):
        if agent_type == AgentType.VANILLA:
            cols = st.columns(2)
            with cols[0]:
                # AU.checkbox_sb_spacer()
                args['sequential'] = st.checkbox('Sequential', value=True)
        else:
            st.write('No additional configs for this agent type.')
    return args