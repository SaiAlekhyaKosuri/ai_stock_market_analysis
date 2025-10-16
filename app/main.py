from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
import pandas as pd
import os
import logging

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Initialize the FastAPI app
# -----------------------------
app = FastAPI(
    title="FinFor Stock Prediction API",
    description="An API to predict stock prices and fetch historical data.",
    version="1.0.0"
)

# -----------------------------
# Add CORS Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Define Absolute Paths for Model/Scaler
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lstm_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.gz')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# -----------------------------
# Load ML Model and Scaler on Startup
# -----------------------------
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}\n"
                     f"Checked paths:\nModel: {MODEL_PATH}\nScaler: {SCALER_PATH}")

# -----------------------------
# Pydantic Model for Prediction Input
# -----------------------------
class StockInput(BaseModel):
    sequence: list[float]

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/api/health", tags=["Health Check"])
def health_check():
    """Simple health check endpoint to confirm the API is running."""
    return {"status": "ok"}

@app.post("/api/predict", tags=["Prediction"])
def predict(stock_input: StockInput):
    """
    Predict the next day's stock price based on a sequence of the
    last 60 days' closing prices.
    """
    logger.info("Prediction endpoint called.")
    if len(stock_input.sequence) != 60:
        raise HTTPException(
            status_code=400,
            detail="Input sequence must contain exactly 60 values."
        )
    try:
        # Reshape and scale the input data
        input_data = np.array(stock_input.sequence).reshape(-1, 1)
        scaled_input = scaler.transform(input_data)
        
        # Reshape for LSTM model (samples, timesteps, features)
        reshaped_input = np.reshape(scaled_input, (1, 60, 1))
        
        # Make prediction
        prediction_scaled = model.predict(reshaped_input)
        
        # Inverse transform the prediction to get the actual price
        prediction = scaler.inverse_transform(prediction_scaled)
        
        logger.info("Prediction successful.")
        return {"prediction": float(prediction[0][0])}
    except Exception as e:
        logger.error(f"An error occurred during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.get("/api/history/{ticker}", tags=["Stock Data"])
def get_history(ticker: str, start_date: str = '2015-01-01', period: str = None):
    """Fetch historical stock data for a given ticker from Yahoo Finance."""
    logger.info(f"Fetching history for ticker: {ticker}")
    try:
        if period:
            data = yf.download(ticker, period=period)
        else:
            data = yf.download(ticker, start=start_date)
        
        if data.empty:
            logger.warning(f"No historical data found for ticker '{ticker}'")
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for ticker '{ticker}'"
            )
        
        # Reset index to make it a column for JSON serialization
        data = data.reset_index()
        logger.info(f"Successfully fetched data for {ticker}.")
        return data.to_dict(orient='records')
        
    except Exception as e:
        logger.error(f"An error occurred while fetching data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching data: {str(e)}")

# -----------------------------
# Serve Frontend
# -----------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def read_root():
    """Serve the main index.html file for the frontend."""
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)