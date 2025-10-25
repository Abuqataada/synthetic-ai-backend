# ============================================
# api/predict_vix75.py
# Institutional AI Predictor for VIX75 (5M/15M)
# Auto-downloads model/scaler/features from GitHub Releases
# ============================================

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import requests
import websocket
from datetime import datetime
from flask import Flask, jsonify, render_template

# ============================================
# FLASK CONFIGURATION
# ============================================
APP = Flask(__name__, template_folder="../templates", static_folder="../static")

DERIV_APP_ID = 1089
DERIV_WS = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
SYMBOL = "R_75"

# ============================================
# MODEL LOCATIONS (GitHub Releases)
# ============================================
MODEL_URL = "https://github.com/Abuqataada/synthetic-ai-backend/releases/download/v1.0/institutional_vix75_model.joblib"
SCALER_URL = "https://github.com/Abuqataada/synthetic-ai-backend/releases/download/v1.0/institutional_scaler.joblib"
FEATURE_URL = "https://github.com/Abuqataada/synthetic-ai-backend/releases/download/v1.0/feature_list.txt"

MODEL_PATH = "institutional_vix75_model.joblib"
SCALER_PATH = "institutional_scaler.joblib"
FEATURE_PATH = "feature_list.txt"

# ============================================
# UTIL: AUTO-DOWNLOAD IF FILES MISSING
# ============================================
def download_if_missing(url, path):
    """Download model/scaler/feature files only if not already present."""
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} from {url} ...")
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Saved {path}")
    else:
        print(f"Found cached file: {path}")

# Ensure model assets exist
download_if_missing(MODEL_URL, MODEL_PATH)
download_if_missing(SCALER_URL, SCALER_PATH)
download_if_missing(FEATURE_URL, FEATURE_PATH)

# Load model/scaler/features
MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
FEATURES = [line.strip() for line in open(FEATURE_PATH).read().splitlines() if line.strip()]

label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

# ============================================
# DATA FETCHING
# ============================================
def fetch_candles(symbol, granularity, count=500):
    """Fetch OHLC candles from Deriv WebSocket API."""
    payload = {
        "ticks_history": symbol,
        "style": "candles",
        "granularity": granularity,
        "count": count,
        "end": "latest"
    }

    data_holder = {}

    def on_message(ws, msg):
        msg = json.loads(msg)
        if "candles" in msg:
            data_holder["candles"] = msg["candles"]
            ws.close()

    def on_open(ws):
        ws.send(json.dumps(payload))

    ws = websocket.WebSocketApp(DERIV_WS, on_open=on_open, on_message=on_message)
    ws.run_forever()

    df = pd.DataFrame(data_holder["candles"])
    df["ts"] = pd.to_datetime(df["epoch"], unit="s")

    # Convert only numeric columns
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df[["ts", "open", "high", "low", "close"]].sort_values("ts").reset_index(drop=True)

# ============================================
# TECHNICAL INDICATORS
# ============================================
def add_indicators(df, prefix=""):
    df = df.copy()
    close = df["close"]

    # EMAs
    df[f"{prefix}ema_10"] = close.ewm(span=10).mean()
    df[f"{prefix}ema_50"] = close.ewm(span=50).mean()
    df[f"{prefix}ema_200"] = close.ewm(span=200).mean()

    # RSI
    delta = close.diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    roll_up, roll_down = up.rolling(14).mean(), down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df[f"{prefix}rsi"] = 100 - (100 / (1 + rs))

    # MACD & Signal
    df[f"{prefix}macd"] = df[f"{prefix}ema_10"] - df[f"{prefix}ema_50"]
    df[f"{prefix}signal"] = df[f"{prefix}macd"].ewm(span=9).mean()

    # Bollinger Bands
    df[f"{prefix}upper_bb"] = close.rolling(20).mean() + 2 * close.rolling(20).std()
    df[f"{prefix}lower_bb"] = close.rolling(20).mean() - 2 * close.rolling(20).std()

    # ATR
    df[f"{prefix}atr"] = (df["high"] - df["low"]).rolling(14).mean()

    # Stochastic Oscillator
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df[f"{prefix}stoch_k"] = 100 * ((close - low14) / (high14 - low14))
    df[f"{prefix}stoch_d"] = df[f"{prefix}stoch_k"].rolling(3).mean()

    return df.dropna()

def merge_timeframes(df5, df15):
    df15 = df15.rename(columns={
        "open": "open_15m", "high": "high_15m",
        "low": "low_15m", "close": "close_15m"
    })
    return pd.merge_asof(df5.sort_values("ts"), df15.sort_values("ts"), on="ts", direction="backward").dropna()

# ============================================
# ROUTES
# ============================================
@APP.route("/")
def index():
    return render_template("index.html")

@APP.route("/predict_vix75")
def predict_vix75():
    """Main prediction endpoint for VIX75 AI model."""
    try:
        df5 = fetch_candles(SYMBOL, 300, 400)
        df15 = fetch_candles(SYMBOL, 900, 400)
    except Exception as e:
        return jsonify({"error": "Failed to fetch candles", "detail": str(e)}), 500

    # Compute indicators for both TFs
    df5 = add_indicators(df5, "5m_")
    df15 = add_indicators(df15, "15m_")

    # Merge 5m + 15m frames
    merged = merge_timeframes(df5, df15)
    if merged.empty:
        return jsonify({"error": "Not enough candle data"}), 400

    # Ensure feature alignment
    for col in FEATURES:
        if col not in merged.columns:
            merged[col] = 0.0
    merged = merged[FEATURES]

    # Scale & predict using latest row
    X = SCALER.transform(merged.iloc[-1:])
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
    else:
        pred = int(MODEL.predict(X)[0])
        probs = [0.0, 0.0, 0.0]
        conf = 1.0

    result = {
        "symbol": SYMBOL,
        "timeframe": "5M",
        "timestamp": int(time.time()),
        "signal": label_map[pred],
        "confidence": round(conf, 4),
        "probabilities": {
            "SELL": float(probs[0]),
            "HOLD": float(probs[1]),
            "BUY": float(probs[2])
        },
        "latest_close": float(merged["close"].iloc[-1])
    }
    return jsonify(result)

# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=False)
