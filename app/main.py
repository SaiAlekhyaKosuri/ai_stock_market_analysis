from fastapi import FastAPI, HTTPException, Request
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
import requests

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
# Define Project Directory Structure
# -----------------------------
# Assuming your directory structure is:
# project_root/
# ├── app/
# │   └── main.py
# ├── models/
# │   └── lstm_model.h5, scaler.gz
# └── static/
#     └── index.html, script.js, style.css
# This setup makes path resolution robust.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lstm_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.gz')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# -----------------------------
# Load ML Model and Scaler on Startup
# -----------------------------
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
    
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Loaded scaler from {SCALER_PATH}")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    raise RuntimeError(f"Could not load ML model or scaler. Ensure they exist at the correct paths.") from e

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
    """Checks the service and its connectivity to Yahoo Finance."""
    try:
        response = requests.get("https://finance.yahoo.com", timeout=5)
        network_status = "healthy" if response.status_code == 200 else "unhealthy"
    except requests.RequestException:
        network_status = "unhealthy"
    return {"status": "ok", "network_to_yahoo": network_status}

@app.post("/api/predict", tags=["Prediction"])
def predict(stock_input: StockInput):
    """Predicts the next day's stock price based on a sequence of 60 closing prices."""
    logger.info("Prediction endpoint called.")
    if len(stock_input.sequence) != 60:
        raise HTTPException(
            status_code=400,
            detail="Input sequence must contain exactly 60 values."
        )
    try:
        input_data = np.array(stock_input.sequence).reshape(-1, 1)
        scaled_input = scaler.transform(input_data)
        reshaped_input = np.reshape(scaled_input, (1, 60, 1))
        
        prediction_scaled = model.predict(reshaped_input)
        prediction = scaler.inverse_transform(prediction_scaled)
        
        logger.info("Prediction successful.")
        return {"prediction": float(prediction[0][0])}
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")

@app.get("/api/history/{ticker}", tags=["Stock Data"])
def get_history(ticker: str, period: str = "1y"):
    """Fetches historical stock data for a given ticker."""
    logger.info(f"Fetching historical data for ticker: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            logger.warning(f"No historical data found for ticker '{ticker}'")
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for ticker '{ticker}'. It might be an invalid symbol."
            )

        # ✅ FIX: Simplified and robust data serialization
        # Reset index to make 'Date' a column and format it correctly.
        data = data.reset_index()
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        
        # Convert DataFrame to a list of dictionaries, which is JSON-friendly.
        # This handles all data types (floats, ints, etc.) automatically.
        response_data = data.to_dict(orient='records')
        
        logger.info(f"Successfully processed {len(response_data)} data points for {ticker}")
        return response_data

    except Exception as e:
        logger.error(f"An error occurred while fetching data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching data from the external provider.")

# -----------------------------
# Serve Frontend
# -----------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

@app.get("/", include_in_schema=False)
def read_root():
    """Serves the main index.html file."""
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

if __name__ == "__main__":
    import uvicorn
    # For production, consider using a different server like Gunicorn with Uvicorn workers.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)