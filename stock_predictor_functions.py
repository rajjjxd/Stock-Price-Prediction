import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Prepare data for ARIMA
def prepare_data_arima(df):
    df.set_index('Date', inplace=True)
    return df['Close']

# Train ARIMA model
def train_arima(data):
    model = ARIMA(data, order=(5,1,0))
    return model.fit()

# Predict future prices using ARIMA
def predict_future_prices_arima(model, last_date, days_to_predict):
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    forecast = model.forecast(steps=days_to_predict)
    return pd.DataFrame({'Date': future_dates, 'Predicted_Price': forecast})

# Prepare data for LSTM
def prepare_data_lstm(df):
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    look_back = 60
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler, look_back

# Train LSTM model
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

# Predict future prices using LSTM
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
    last_date = df.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    return pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices.flatten()})

# Plot results
def plot_results(df, future_df, ticker, model_name, ax):
    ax.clear()
    ax.plot(df['Date'], df['Close'], label='Historical Prices')
    ax.plot(future_df['Date'], future_df['Predicted_Price'], label='Predicted Prices', linestyle='--')
    ax.set_title(f'{ticker} Stock Price Prediction ({model_name})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

# Analyze prediction results
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

# Tkinter GUI for Stock Prediction
class StockPredictorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Stock Price Predictor")
        self.master.geometry("800x800")

        self.stocks = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "FB", "TSLA", "JPM", "WMT", "DIS", "NFLX",
            "SPY", "QQQ", "VTI", "IWM", "BABA", "TSM", "TM", "SONY", "COIN", "MARA"
        ]

        self.create_widgets()

    def create_widgets(self):
        # Stock selection
        tk.Label(self.master, text="Select Stocks:").grid(row=0, column=0, padx=5, pady=5)
        self.stock_var = tk.StringVar()
        self.stock_dropdown = ttk.Combobox(self.master, textvariable=self.stock_var, values=self.stocks, state="readonly")
        self.stock_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.stock_dropdown.set(self.stocks[0])

        # Date selection
        tk.Label(self.master, text="Start Date:").grid(row=1, column=0, padx=5, pady=5)
        self.start_date = DateEntry(self.master, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.start_date.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.master, text="End Date:").grid(row=2, column=0, padx=5, pady=5)
        self.end_date = DateEntry(self.master, width=12, background='darkblue', foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
        self.end_date.grid(row=2, column=1, padx=5, pady=5)

        # Days to predict
        tk.Label(self.master, text="Days to Predict:").grid(row=3, column=0, padx=5, pady=5)
        self.days_var = tk.StringVar(value="15")
        self.days_entry = tk.Entry(self.master, textvariable=self.days_var)
        self.days_entry.grid(row=3, column=1, padx=5, pady=5)

        # Model selection
        tk.Label(self.master, text="Select Model:").grid(row=4, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar(value="LSTM")
        self.model_dropdown = ttk.Combobox(self.master, textvariable=self.model_var, values=["ARIMA", "LSTM"], state="readonly")
        self.model_dropdown.grid(row=4, column=1, padx=5, pady=5)

        # Predict button
        self.predict_button = tk.Button(self.master, text="Predict", command=self.predict)
        self.predict_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

        # Analysis text widget
        self.analysis_text = tk.Text(self.master, height=10, width=60)
        self.analysis_text.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

    def predict(self):
        ticker = self.stock_var.get()
        start_date = self.start_date.get_date()
        end_date = self.end_date.get_date()
        days_to_predict = int(self.days_var.get())
        model_choice = self.model_var.get()

        df = fetch_stock_data(ticker, start_date, end_date)

        if model_choice == 'ARIMA':
            data = prepare_data_arima(df)
            model = train_arima(data)
            future_df = predict_future_prices_arima(model, df['Date'].iloc[-1], days_to_predict)
        elif model_choice == 'LSTM':
            X, y, scaler, look_back = prepare_data_lstm(df)
            model = train_lstm(X, y)
            future_df = predict_future_prices_lstm(model, df, scaler, days_to_predict, look_back)

        plot_results(df, future_df, ticker, model_choice, self.ax)
        self.canvas.draw()

        # Generate and display analysis
        analysis = analyze_prediction(df, future_df)
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert(tk.END, analysis)

        print("Future price predictions:")
        print(future_df)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()