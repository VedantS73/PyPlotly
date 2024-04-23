import webbrowser
import streamlit as st
from stock_data import stock_names_symbols
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Tickertape Webhooks (LSTM)",
    page_icon="🛬"
)

symbol = None

st.title("Stock Price Prediction")
st.sidebar.success("Select a page above.")

@st.cache_data
def load_stock_data(symbol):
    """
    Function to load stock data from CSV file based on symbol.
    """
    file_path = f"data/{symbol}.csv"
    return pd.read_csv(file_path)

stocks = stock_names_symbols()
# Dropdown with options
selected_stock = st.selectbox(
    "Select a stock:",
    [f"{stock_name} ({symbol[:-4]})" for stock_name, symbol in stock_names_symbols()]
)

if selected_stock:
    symbol = selected_stock.split(" ")[-1][1:-1]  # Extract symbol from selected stock

    # Load stock data (cached for better performance)
    stock_data = load_stock_data(symbol)
    
    # Plot stock data
    fig = px.line(stock_data, x='Date', y='Close', title=f"Stock Chart for {selected_stock}")
    st.plotly_chart(fig)
    
    open_button = st.button("Open in Web", key="web", help="Click to redirect")

    # Plot raw data
    oned_returns = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]
    onew_returns = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-1]) / stock_data['Close'].iloc[-1]
    onem_returns = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-30]) / stock_data['Close'].iloc[-30]
    oney_returns = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-252]) / stock_data['Close'].iloc[-252]

    oned_returns = round(oned_returns * 100, 2)
    onew_returns = round(onew_returns * 100, 2)
    onem_returns = round(onem_returns * 100, 2)
    oney_returns = round(oney_returns * 100, 2)

    st.write(f"1 Day returns: {oned_returns}%")
    st.write(f"1 Week returns: {onew_returns}%")
    st.write(f"30 Day returns: {onem_returns}%")
    st.write(f"1 Year returns: {oney_returns}%")

if selected_stock and open_button:
    # website = f"https://in.tradingview.com/symbols/NSE-{symbol}/"
    website = f"https://in.tradingview.com/chart/jOuwKUuV/?symbol=NSE%3A{symbol}"
    st.write(f"Opening {website}")
    st.write("Please wait...")
    webbrowser.open(website)