import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import date

st.set_page_config(
    page_title="StockAI - LSTM Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #464b59;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    /* Hide Streamlit Default Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("🎛️ Settings")
st.sidebar.subheader("Input Parameters")

ticker = st.sidebar.text_input("Stock Ticker", "GOOGL")
start_date = st.sidebar.date_input("Start Date", date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

if st.sidebar.button("Run Prediction"):
    st.session_state.run = True
else:
    if 'run' not in st.session_state:
        st.session_state.run = False

@st.cache_data
def load_data(symbol, start, end):
    try:
        # Downloading data
        data = yf.download(symbol, start=start, end=end)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

st.title("📈 Stock Price Prediction System")
st.markdown(f"### powered by **LSTM Neural Networks**")

if st.session_state.run:
    with st.spinner('Fetching market data...'):
        df = load_data(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
        st.stop()

    try:
        latest_close = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2])
        daily_change = latest_close - prev_close
        pct_change = (daily_change / prev_close) * 100
        high_val = float(df['High'].max())
        low_val = float(df['Low'].min())
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Current Price", value=f"${latest_close:.2f}")
    with col2:
        st.metric(label="Daily Change", value=f"{daily_change:.2f}", delta=f"{pct_change:.2f}%")
    with col3:
        st.metric(label="Highest (Period)", value=f"${high_val:.2f}")
    with col4:
        st.metric(label="Lowest (Period)", value=f"${low_val:.2f}")
    
    st.markdown("---")

    tab1, tab2 = st.tabs(["📊 Market Overview", "🧠 LSTM Prediction"])
    with tab1:
        st.subheader(f"Historical Price Analysis: {ticker}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='#00CC96')))
        fig.layout.update(title_text=f'{ticker} Time Series Data', xaxis_rangeslider_visible=True, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

 
        st.subheader("Technical Indicators: Moving Averages")
        close_prices = pd.to_numeric(df['Close'], errors='coerce')
        ma100 = close_prices.rolling(100).mean()
        ma200 = close_prices.rolling(200).mean()

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=close_prices, mode='lines', name='Close Price', line=dict(color='blue', width=1)))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=ma100, mode='lines', name='100-Day MA', line=dict(color='orange', width=2)))
        fig_ma.add_trace(go.Scatter(x=df['Date'], y=ma200, mode='lines', name='200-Day MA', line=dict(color='green', width=2)))
        fig_ma.update_layout(template="plotly_dark", title="100 vs 200 Day Moving Average Trend")
        st.plotly_chart(fig_ma, use_container_width=True)
    with tab2:
        st.subheader("LSTM Model Prediction Results")
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

        scaler = MinMaxScaler(feature_range=(0,1))
        
        if len(data_training) == 0 or len(data_testing) == 0:
            st.error("Not enough data to perform prediction. Please select a longer date range.")
        else:
            data_training_array = scaler.fit_transform(data_training)

            # Load Model
            try:
                model = load_model('keras_model.h5')
                
                # Prepare Test Data
                past_100_days = data_training.tail(100)
                final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                input_data = scaler.fit_transform(final_df)

                x_test = []
                y_test = []

                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i-100: i])
                    y_test.append(input_data[i, 0])

                x_test, y_test = np.array(x_test), np.array(y_test)
                
                # Prediction
                with st.spinner('Running LSTM Model inference...'):
                    y_predicted = model.predict(x_test)

                # Inverse Scaling
                scale_factor = 1/scaler.scale_[0]
                y_predicted = y_predicted * scale_factor
                y_test = y_test * scale_factor

                # Prediction vs Actual Chart
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(y=y_test.flatten(), mode='lines', name='Original Price', line=dict(color='cyan')))
                fig_pred.add_trace(go.Scatter(y=y_predicted.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
                fig_pred.update_layout(
                    title="Predicted vs Actual Closing Prices",
                    xaxis_title="Time (Days)",
                    yaxis_title="Price",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            except Exception as e:
                st.error(f"Error loading model or processing data: {str(e)}")
                st.warning("Ensure 'keras_model.h5' is in the directory and matches the input shape.")

else:
    st.info(" Please enter a stock ticker and click 'Run Prediction' to begin.")