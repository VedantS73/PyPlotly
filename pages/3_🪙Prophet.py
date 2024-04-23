# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Price Prediction (Prophet)')

stocks = (
    'GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'TSLA', 'FB', 'NFLX', 'NVDA',
    'DIS', 'CRM', 'BA', 'IBM', 'PYPL', 'SBUX', 'INTC', 'AMD', 'CSCO',
    'QCOM', 'ORCL', 'UBER', 'LYFT', 'WMT', 'JNJ', 'PG', 'KO', 'PEP', 'XOM',
    'CVX', 'V', 'MA', 'JPM', 'GS', 'BAC', 'C', 'WFC', 'HD', 'LOW', 'MCD'
    # Add more symbols as needed
)
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

if selected_stock:
    train_button = st.button("Predict Stock Movement with Prophet", key="train_button", help="Click to train")

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

if train_button:
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    oned_returns = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
    onew_returns = (data['Close'].iloc[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
    onem_returns = (data['Close'].iloc[-1] - data['Close'].iloc[-30]) / data['Close'].iloc[-30]
    oney_returns = (data['Close'].iloc[-1] - data['Close'].iloc[-252]) / data['Close'].iloc[-252]

    oned_returns = round(oned_returns * 100, 2)
    onew_returns = round(onew_returns * 100, 2)
    onem_returns = round(onem_returns * 100, 2)
    oney_returns = round(oney_returns * 100, 2)

    st.write(f"1 Day returns: {oned_returns}%")
    st.write(f"1 Week returns: {onew_returns}%")
    st.write(f"30 Day returns: {onem_returns}%")
    st.write(f"1 Year returns: {oney_returns}%")

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)