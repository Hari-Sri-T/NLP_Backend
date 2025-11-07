import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# -----------------------------------------------------------------
# REMOVED: from tensorflow import keras
# ADDED: The TFLite interpreter
import tflite_runtime.interpreter as tflite
# -----------------------------------------------------------------
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

def predict_next_close(df):
    """
    Uses the TFLITE model to predict next closing price.
    (This function is UPDATED)
    """
    X, scaler = preprocess_data(df)
    
    # -----------------------------------------------------------------
    # --- Start of TFLite Prediction ---
    
    # 1. Load the TFLite model
    interpreter = tflite.Interpreter(model_path=os.path.abspath(TFLITE_MODEL_PATH))
    interpreter.allocate_tensors()

    # 2. Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Set the input tensor (must be float32)
    # The X from preprocess_data is (1, 60, 5), which is correct
    interpreter.set_tensor(input_details[0]['index'], np.array(X, dtype=np.float32))

    # 4. Run inference
    interpreter.invoke()

    # 5. Get the prediction
    pred = interpreter.get_tensor(output_details[0]['index'])
    
    # --- End of TFLite Prediction ---
    # -----------------------------------------------------------------
    
    # Inverse transform (this logic is the same as before)
    dummy = np.zeros((1, len(FEATURES)))
    dummy[:, 0] = pred.ravel()
    predicted_close = scaler.inverse_transform(dummy)[:, 0][0]
    
    return float(predicted_close)