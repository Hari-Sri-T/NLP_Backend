import yfinance as yf
import numpy as np
import pandas as pd

import requests
import logging

from sklearn.preprocessing import MinMaxScaler
import os

# -----------------------------------------------------------------
# REMOVED: The global model loading
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_multivariate_lstm.keras")
# model = keras.models.load_model(os.path.abspath(MODEL_PATH))
# -----------------------------------------------------------------

# ADDED: Path to the new TFLite model
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.tflite")

FEATURES = ["close", "open", "high", "low", "volume"]
TIME_STEP = 60

def get_stock_data(symbol="GOOG"):
    """
    Fetches last 90 days of stock data from Yahoo Finance.
    (This function is unchanged)
    """
    df = yf.download(symbol, period="90d", interval="1d")
    df = df.rename(columns={
        "Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"
    })
    return df[FEATURES]

def preprocess_data(df):
    """
    Scales data and returns last TIME_STEP days in correct shape.
    (This function is unchanged)
    """
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(df)
    X = scaled[-TIME_STEP:]
    return np.expand_dims(X, axis=0), scaler

def predict_next_close(symbol):
    """
    Calls the Hugging Face AI service to get the predicted close price.
    Returns the predicted price as a float, or None if it fails.
    """
    # This is the URL to your existing Hugging Face API endpoint
    hf_url = "https://harisri-ai-stock-advisor.hf.space/predict"
    
    try:
        # 30-second timeout to handle the HF Space waking up
        response = requests.get(f"{hf_url}?symbol={symbol}", timeout=30)
        
        if response.status_code == 200:
            prediction_data = response.json()
            # Check if the key exists before returning
            if "predicted_close" in prediction_data:
                return float(prediction_data["predicted_close"])
            else:
                logging.warning(f"AI service response missing 'predicted_close' key: {prediction_data}")
                return None
        else:
            # Log if the server didn't return 200 OK
            logging.warning(f"AI service returned status {response.status_code}")
            return None
            
    except requests.RequestException as e:
        # If the HF space is asleep or fails, log the error
        logging.error(f"AI service call failed: {e}")
        return None