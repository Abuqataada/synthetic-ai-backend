# ===========================================================
# vix75_data_pipeline.py
# ===========================================================
import websocket
import json
import pandas as pd
import time
from datetime import datetime, timezone
import numpy as np
import os

# ---------------- CONFIG ----------------
APP_ID = 1089
SYMBOL = "R_75"
SEGMENT_SIZE = 5000
DAYS_BACK = 365
SAVE_DIR = "data"
WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
GRANULARITIES = {"5m": 300, "15m": 900}

os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================
# 1. WebSocket candle fetcher
# ===========================================================
def fetch_segment(end_time, granularity):
    payload = {
        "ticks_history": SYMBOL,
        "style": "candles",
        "granularity": granularity,
        "count": SEGMENT_SIZE,
        "end": str(int(end_time.timestamp()))
    }
    candles = []

    def on_message(ws, message):
        msg = json.loads(message)
        if 'candles' in msg:
            candles.extend(msg['candles'])
            ws.close()

    def on_open(ws):
        ws.send(json.dumps(payload))

    ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message)
    ws.run_forever()
    return candles


def collect_full_data(granularity_name, granularity_seconds, days_back):
    all_candles = []
    end_time = datetime.now(tz=timezone.utc)
    total_segments = int((days_back * 24 * 60 / (granularity_seconds / 60)) / SEGMENT_SIZE) + 1

    print(f"\nFetching {granularity_name} data for {SYMBOL} ({days_back} days)...")

    for i in range(total_segments):
        print(f"     Segment {i+1}/{total_segments} ending {end_time}")
        candles = fetch_segment(end_time, granularity_seconds)
        if not candles:
            break
        all_candles.extend(candles)
        first_ts = candles[0]['epoch']
        end_time = datetime.fromtimestamp(first_ts - granularity_seconds, tz=timezone.utc)
        time.sleep(1.2)

    df = pd.DataFrame(all_candles)
    df.rename(columns={'epoch': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df[['timestamp', 'open', 'high', 'low', 'close']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.to_csv(os.path.join(SAVE_DIR, f"vix75_{granularity_name}.csv"), index=False)
    print(f"Saved {len(df)} rows to data/vix75_{granularity_name}.csv")
    return df


# ===========================================================
# 2. Indicator Functions
# ===========================================================
def add_indicators(df, prefix=""):
    df = df.copy()
    close = df['close']

    df[f'{prefix}rsi'] = close.diff().apply(lambda x: 0 if pd.isna(x) else x)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df[f'{prefix}rsi'] = 100 - (100 / (1 + rs))

    df[f'{prefix}ema_10'] = close.ewm(span=10, adjust=False).mean()
    df[f'{prefix}ema_50'] = close.ewm(span=50, adjust=False).mean()
    df[f'{prefix}ema_200'] = close.ewm(span=200, adjust=False).mean()

    df[f'{prefix}macd'] = df[f'{prefix}ema_10'] - df[f'{prefix}ema_50']
    df[f'{prefix}signal'] = df[f'{prefix}macd'].ewm(span=9, adjust=False).mean()

    df[f'{prefix}upper_bb'] = close.rolling(20).mean() + 2 * close.rolling(20).std()
    df[f'{prefix}lower_bb'] = close.rolling(20).mean() - 2 * close.rolling(20).std()

    df[f'{prefix}atr'] = (df['high'] - df['low']).rolling(14).mean()
    df[f'{prefix}stoch_k'] = ((close - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
    df[f'{prefix}stoch_d'] = df[f'{prefix}stoch_k'].rolling(3).mean()

    return df


# ===========================================================
# 3. Merge and Enrich
# ===========================================================
def merge_timeframes(df_5m, df_15m):
    df_15m = df_15m.rename(columns={
        'open': 'open_15m', 'high': 'high_15m', 'low': 'low_15m', 'close': 'close_15m'
    })

    merged = pd.merge_asof(
        df_5m.sort_values('timestamp'),
        df_15m.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    merged.dropna(inplace=True)
    return merged


if __name__ == "__main__":
    print("=== Institutional VIX75 Data Pipeline ===")
    df_5m = collect_full_data("5m", GRANULARITIES["5m"], DAYS_BACK)
    df_15m = collect_full_data("15m", GRANULARITIES["15m"], DAYS_BACK)

    print("\nComputing indicators...")
    df_5m = add_indicators(df_5m, prefix="5m_")
    df_15m = add_indicators(df_15m, prefix="15m_")

    print("\nMerging and enriching dataset...")
    merged = merge_timeframes(df_5m, df_15m)
    merged.to_csv(os.path.join(SAVE_DIR, "vix75_merged_enriched.csv"), index=False)
    print(f"Final enriched dataset saved -> data/vix75_merged_enriched.csv")
    print(f"Total rows: {len(merged)}")
