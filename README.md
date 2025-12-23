# 📈 StockAI - LSTM Predictor

A web-based application that predicts stock prices using Long Short-Term Memory (LSTM) neural networks. Built with **Streamlit**, this tool allows users to visualize historical stock data, analyze technical indicators, and compare predicted prices against actual market trends.

## 🚀 Features

* **Real-time Data Fetching**: Retrieves stock market data dynamically using `yfinance`.
* **Interactive Dashboard**:
    * Visualizes Time Series data with interactive Plotly charts.
    * Displays key metrics: Current Price, Daily Change, and High/Low values.
* **Technical Analysis**: Plots 100-day and 200-day Moving Averages (MA) to identify trends.
* **LSTM Prediction**: Uses a pre-trained Keras model to forecast stock prices and compares them with actual values.

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **ML & Data Processing**: TensorFlow (Keras), Scikit-Learn, NumPy, Pandas
* **Visualization**: Plotly Graph Objects
* **Data Source**: Yahoo Finance (`yfinance`)

## 📂 Project Structure

```text
├── app.py                # Main Streamlit application
├── train_model.py        # Script to train and save the LSTM model
├── keras_model.h5        # Pre-trained LSTM model file
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
