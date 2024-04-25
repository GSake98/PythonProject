import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM, Dropout # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
import yfinance as yf
from ta import add_all_ta_features  # Ensure 'ta' is installed
import os
import time
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pytz

# API keys setup (make sure to replace these with your own keys or environment variables)
os.environ['APCA_API_KEY_ID'] = 'PKJMZ0P3VIK5NND712VA'
os.environ['APCA_API_SECRET_KEY'] = 'M4ayzNbXH6X4CJQQfcClsALLAsQbKr50KxdyOFAN'
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(os.environ['APCA_API_KEY_ID'], os.environ['APCA_API_SECRET_KEY'], os.environ['APCA_API_BASE_URL'], api_version='v2')

def fetch_data(symbol, period="7d", interval="1m"):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return df

def preprocess_data(df, sequence_length=5):
    features = df[['Close', 'volume_adi', 'momentum_rsi', 'trend_macd', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl']].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(scaled_features[i, 0])
    return np.array(X), np.array(y), scaler

def build_and_train_model(X_train, y_train, X_val, y_val, lstm_units=50, epochs=10, batch_size=32, dropout_rate=0.2):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions_inversed = scaler.inverse_transform(np.hstack([predictions, np.zeros((predictions.shape[0], 6))]))[:, 0]
    y_test_inversed = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 6))]))[:, 0]
    
    mse = mean_squared_error(y_test_inversed, predictions_inversed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inversed, predictions_inversed)
    mape = np.mean(np.abs((y_test_inversed - predictions_inversed) / y_test_inversed)) * 100
    r2 = r2_score(y_test_inversed, predictions_inversed)
    success_rate = 100 - mape
    print(f"Model evaluation completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}, Success Rate: {success_rate:.2f}%")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "Success Rate": success_rate}

def predict(model, X_input, scaler):
    predicted_data = model.predict(X_input)
    dummy_features = np.zeros((predicted_data.shape[0], 6))
    predicted_full = np.hstack([predicted_data, dummy_features])
    predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
    return predicted_prices

def post_process_and_visualize(actual_prices, predicted_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='red')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    

def fetch_current_price(symbol):
    """
    Fetches the current price for the given `symbol`.
    """
    try:
        latest_trade = api.get_latest_trade(symbol)
        print(f"Current Price: {latest_trade.price}")
        return latest_trade.price
    except tradeapi.rest.APIError as e:
        print(f"APIError: {e}")
        return None

def predict_prices(initial_price):
    """
    Generate predictions for the next 60 minutes from the initial price.
    Placeholder for the prediction model.
    """
    predictions = [initial_price * (1 + 0.0001 * i) for i in range(3)]  # Example prediction logic
    return predictions

def is_successful_prediction(predicted, actual, threshold=0.001):
    """
    Determines if the prediction is successful within a certain percentage threshold.
    """
    return abs(predicted - actual) / actual <= threshold 

def evaluate_predictions(predictions, actual_prices):
    """
    Evaluates the predictions against actual prices and calculates the success rate.
    """
    successes = [is_successful_prediction(pred, act) for pred, act in zip(predictions, actual_prices)]
    success_rate = np.mean(successes)
    print(f"Success Rate: {success_rate * 100:.2f}%")
    return success_rate

def retrain_model():
    """
    Placeholder function for retraining the model.
    """
    print("Retraining the model...")
def fetch_and_compare_prices(symbol):
    initial_price = fetch_current_price(symbol)
    if initial_price is None:
        return
    
    predictions = predict_prices(initial_price)
    
    actual_prices = []
    prediction_times = []
    
    for i in range(3):  # Fetch prices every minute for the next hour
        time.sleep(60)  # Wait a minute
        current_price = fetch_current_price(symbol)
        if current_price is not None:
            actual_prices.append(current_price)
            prediction_times.append(datetime.now(pytz.timezone('America/New_York')))
    
    # Store or print comparison results
    for pred, act, pred_time in zip(predictions, actual_prices, prediction_times):
        print(f"Time: {pred_time}, Predicted: {pred}, Actual: {act}")
    success_rate = evaluate_predictions(predictions, actual_prices)
    if success_rate < 0.996:
        retrain_model()
        
def full_workflow(symbol):
    df = fetch_data(symbol)
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test_final, y_val, y_test_final = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    model, history = build_and_train_model(X_train, y_train, X_val, y_val)
    evaluation_results = evaluate_model(model, X_test_final, y_test_final, scaler)
    actual_prices = scaler.inverse_transform(np.hstack([y_test_final.reshape(-1, 1), np.zeros((y_test_final.shape[0], 6))]))[:, 0]
    predicted_prices = predict(model, X_test_final, scaler)
    post_process_and_visualize(actual_prices, predicted_prices)
    model.save("F:\\PytonTradingBot\\trained_model.h5")
    print("High-performing model saved.")
    fetch_current_price(symbol)
    fetch_and_compare_prices(symbol)
    

if __name__ == "__main__":
    full_workflow('GH')
