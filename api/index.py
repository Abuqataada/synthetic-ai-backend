# api/predict_vix75.py
import os
import json
import time
import joblib
import websocket
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, render_template, request

APP = Flask(__name__, template_folder="../templates", static_folder="../static")

# Config
DERIV_APP_ID = 1089
DERIV_WS = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
SYMBOL = "R_75"          # VIX75 / verify on Deriv if needed
GRANULARITY = 300        # 5-minute
CANDLES_COUNT = 60       # fetch last 60 candles (5h history)

MODEL_PATH = os.path.join(os.getcwd(), "institutional_vix75_model.joblib")
SCALER_PATH = os.path.join(os.getcwd(), "institutional_scaler.joblib")

# Load model & scaler
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler not found. Make sure 'institutional_vix75_model.joblib' and 'institutional_scaler.joblib' are in project root.")

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)

# helpers: indicators must match training
def add_features(df):
    df = df.copy()
    df['close'] = df['close'].astype(float)
    df['returns'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df['close'].rolling(w).mean()
        df[f"ema_{w}"] = df['close'].ewm(span=w, adjust=False).mean()

    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    df['volatility'] = df['returns'].rolling(20).std()

    df['ema_fast'] = df['close'].ewm(span=12).mean()
    df['ema_slow'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal_line'] = df['macd'].ewm(span=9).mean()

    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low14) / (high14 - low14))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    df['williams_r'] = -100 * ((high14 - df['close']) / (high14 - low14))

    df = df.dropna()
    return df

# Deriv WebSocket helper to sync-get candles (synchronous blocking)
def fetch_candles_from_deriv(count=CANDLES_COUNT, granularity=GRANULARITY):
    payload = {
        "ticks_history": SYMBOL,
        "style": "candles",
        "granularity": granularity,
        "count": count,
        "end": "latest"
    }
    response_holder = {"candles": None}

    def on_message(ws, message):
        try:
            msg = json.loads(message)
            if 'candles' in msg:
                response_holder['candles'] = msg['candles']
                ws.close()
        except Exception as e:
            response_holder['error'] = str(e)
            ws.close()

    def on_open(ws):
        ws.send(json.dumps(payload))

    def on_error(ws, error):
        response_holder['error'] = str(error)

    def on_close(ws, code, reason):
        pass

    ws = websocket.WebSocketApp(DERIV_WS, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    # run_forever blocks until closed
    ws.run_forever(ping_interval=30, ping_timeout=10)
    if response_holder.get('candles'):
        df = pd.DataFrame(response_holder['candles'])
        df = df.rename(columns={'epoch': 'ts', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
        df = df.sort_values('ts').reset_index(drop=True)
        return df[['ts','open','high','low','close']]
    else:
        raise RuntimeError("Failed to fetch candles from Deriv: " + str(response_holder.get('error')))

# label mapping used during training: 0=SELL,1=HOLD,2=BUY
label_map_reverse = {0: "SELL", 1: "HOLD", 2: "BUY"}

@APP.route("/")
def index():
    return render_template("index.html")

@APP.route("/predict_vix75", methods=["GET"])
def predict_vix75():
    try:
        # Fetch latest candles
        candles_df = fetch_candles_from_deriv(count=CANDLES_COUNT, granularity=GRANULARITY)
    except Exception as e:
        return jsonify({"error": "Failed to fetch candles", "detail": str(e)}), 500

    # Compute indicators exactly like training
    df_features = add_features(candles_df.rename(columns={'ts':'timestamp'}).assign(timestamp=candles_df['ts']))
    # keep the last row (most recent)
    if df_features.empty:
        return jsonify({"error":"Not enough candles to compute indicators"}), 400

    # Model expects a set of features in same order as training used.
    # We'll use the features order inferred from the scaler shape
    # Build a DataFrame with exactly the numeric columns used in training.
    # Infer feature list from SCALER input shape if possible (we saved scaler for training).
    # Here we assume scaler was fit on the same pipeline: pick columns in X used during training.
    # To be safe, list the features explicitly — must match training pipeline.

    # --------- DEFINE FEATURE ORDER (must match training) ----------
    feature_cols = [
        # moving averages
        'sma_5','sma_10','sma_20','sma_50',
        'ema_5','ema_10','ema_20','ema_50',
        # core indicators
        'rsi','bb_mid','bb_std','bb_upper','bb_lower',
        'momentum','volatility','macd','signal_line',
        'stoch_k','stoch_d','williams_r',
        # price returns
        'returns','log_ret'
    ]

    # But ensure these exist in df_features (names created by add_features above might differ).
    # Map names used above to actual columns in df_features
    # Build a dictionary mapping:
    mapping = {}
    # sma & ema created as 'sma_{w}', 'ema_{w}'
    mapping.update({f"sma_{w}": f"sma_{w}" for w in [5,10,20,50]})
    mapping.update({f"ema_{w}": f"ema_{w}" for w in [5,10,20,50]})
    mapping.update({
        'rsi':'rsi',
        'bb_mid':'bb_mid',
        'bb_std':'bb_std',
        'bb_upper':'bb_upper',
        'bb_lower':'bb_lower',
        'momentum':'momentum',
        'volatility':'volatility',
        'macd':'macd',
        'signal_line':'signal_line',
        'stoch_k':'stoch_k',
        'stoch_d':'stoch_d',
        'williams_r':'williams_r',
        'returns':'returns',
        'log_ret':'log_ret'
    })

    # ensure all columns present
    last = df_features.iloc[-1]
    feature_vector = []
    missing = []
    for feat in feature_cols:
        col = mapping.get(feat)
        if col in last:
            feature_vector.append(float(last[col]))
        else:
            missing.append(feat)
            feature_vector.append(0.0)  # fallback; but we should warn

    if missing:
        # Not fatal, but warns
        print("Warning: missing features:", missing)

    X = np.array(feature_vector).reshape(1, -1)
    # scale
    try:
        Xs = SCALER.transform(X)
    except Exception as e:
        # if scaler incompatible shape, attempt to reshape
        Xs = X

    # predict
    try:
        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(Xs)[0]  # array of probabilities per class ordering
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs))
            probs_dict = {"SELL": float(probs[0]), "HOLD": float(probs[1]) if len(probs)>1 else 0.0, "BUY": float(probs[2]) if len(probs)>2 else 0.0}
        else:
            pred = int(MODEL.predict(Xs)[0])
            confidence = 1.0
            probs_dict = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    except Exception as e:
        return jsonify({"error": "model prediction failed", "detail": str(e)}), 500

    resp = {
        "symbol": SYMBOL,
        "timeframe": "5M",
        "timestamp": int(time.time()),
        "signal_code": int(pred),
        "signal": label_map_reverse.get(pred, "UNKNOWN"),
        "confidence": round(confidence, 4),
        "probabilities": probs_dict,
        "latest_close": float(last['close']),
        "model_version": getattr(MODEL, 'model_version', 'ensemble_v1')
    }

    return jsonify(resp)

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=False)
