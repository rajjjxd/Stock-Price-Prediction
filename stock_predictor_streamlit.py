import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Functions for ARIMA model
def prepare_data_arima(df):
    return df.set_index('Date')['Close']

def train_arima(data):
    model = ARIMA(data, order=(5,1,0))
    return model.fit()

def predict_future_prices_arima(model, last_date, days_to_predict):
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    forecast = model.forecast(steps=days_to_predict)
    return pd.DataFrame({'Date': future_dates, 'Predicted_Price': forecast})

# Functions for LSTM model
def prepare_data_lstm(df):
    data = df[['Date', 'Close']].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler, look_back

def train_lstm(X, y):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=1, epochs=1)
    return model

def predict_future_prices_lstm(model, df, scaler, days_to_predict, look_back):
    last_60_days = df['Close'].values[-look_back:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    future_prices = []
    for _ in range(days_to_predict):
        predicted_price = model.predict(X_test)
        future_prices.append(predicted_price[0, 0])
        X_test = np.append(X_test[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
    
    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    return pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices.flatten()})

# Function to plot results
def plot_results(df, future_df, ticker, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Close'], label='Historical Prices')
    ax.plot(future_df['Date'], future_df['Predicted_Price'], label='Predicted Prices', linestyle='--')
    ax.set_title(f'{ticker} Stock Price Prediction ({model_name})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    return fig

# Function to analyze prediction
def analyze_prediction(df, future_df):
    last_price = df['Close'].iloc[-1]
    future_prices = future_df['Predicted_Price']
    max_price = future_prices.max()
    max_price_day = future_prices.idxmax()
    days_to_max = (future_df.loc[max_price_day, 'Date'] - df['Date'].iloc[-1]).days
    expected_return = (max_price - last_price) / last_price * 100

    min_price = future_prices.min()
    min_price_day = future_prices.idxmin()
    days_to_min = (future_df.loc[min_price_day, 'Date'] - df['Date'].iloc[-1]).days

    analysis = f"Last closing price: ${last_price:.2f}\n\n"
    analysis += f"Maximum predicted price: ${max_price:.2f} in {days_to_max} days\n"
    analysis += f"Expected return at maximum: {expected_return:.2f}%\n\n"
    analysis += f"Minimum predicted price: ${min_price:.2f} in {days_to_min} days\n\n"

    if days_to_max > days_to_min:
        analysis += "Recommendation: Consider short-term investment or waiting for a dip before investing.\n"
        analysis += f"Potential strategy: Buy at or near ${min_price:.2f} and sell at or near ${max_price:.2f}."
    else:
        analysis += "Recommendation: Consider immediate investment with a short-term horizon.\n"
        analysis += f"Potential strategy: Buy now and aim to sell at or near ${max_price:.2f}."

    return analysis

# Function to run prediction
def run_prediction(ticker, start_date, end_date, days_to_predict, model_choice):
    df = fetch_stock_data(ticker, start_date, end_date)

    if model_choice == 'ARIMA':
        data = prepare_data_arima(df)
        model = train_arima(data)
        future_df = predict_future_prices_arima(model, df['Date'].iloc[-1], days_to_predict)
    elif model_choice == 'LSTM':
        X, y, scaler, look_back = prepare_data_lstm(df)
        model = train_lstm(X, y)
        future_df = predict_future_prices_lstm(model, df, scaler, days_to_predict, look_back)

    return df, future_df

# Streamlit app
def main():
    st.set_page_config(page_title="Stock Price Predictor", layout="wide")
    st.title("Stock Price Predictor")

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")

    stocks = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "FB", "TSLA", "JPM", "WMT", "DIS", "NFLX",
        "SPY", "QQQ", "VTI", "IWM", "BABA", "TSM", "TM", "SONY", "COIN", "MARA"
    ]
    
    ticker = st.sidebar.selectbox("Select Stock", stocks)
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    days_to_predict = st.sidebar.number_input("Days to Predict", min_value=1, max_value=365, value=15)
    model_choice = st.sidebar.selectbox("Select Model", ["ARIMA", "LSTM"])

    if st.sidebar.button("Predict"):
        with st.spinner("Predicting..."):
            df, future_df = run_prediction(ticker, start_date, end_date, days_to_predict, model_choice)

        # Plot results
        st.subheader("Stock Price Prediction Chart")
        fig = plot_results(df, future_df, ticker, model_choice)
        st.pyplot(fig)

        # Display analysis
        st.subheader("Prediction Analysis")
        analysis = analyze_prediction(df, future_df)
        st.text(analysis)

        # Display future price predictions
        st.subheader("Future Price Predictions")
        st.dataframe(future_df)

if __name__ == "__main__":
    main()