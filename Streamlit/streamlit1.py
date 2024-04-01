import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
         # Simple Stock Price App
         
         Shown are the stock closing price and volume of APPLE!
         
         """)

tickerSymbol = "AAPL"

tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period = '1d', start = '2010-5-31', end = '2024-3-24')

st.write("""
         ## Closing Price
         """)
st.line_chart(tickerDf.Close)
st.write("""
         ## Volume
         """)
st.line_chart(tickerDf.Volume)
