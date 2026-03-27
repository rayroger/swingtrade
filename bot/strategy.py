import pandas as pd

def compute_indicators(df, sma_short=10, sma_long=50, momentum_period=10):
    df["SMA_short"] = df["close"].rolling(sma_short).mean()
    df["SMA_long"] = df["close"].rolling(sma_long).mean()
    df["momentum"] = df["close"].pct_change(momentum_period)
    return df

def trend_score(last_row):
    # Higher score = stronger trend
    return last_row["momentum"] * (last_row["SMA_short"] - last_row["SMA_long"]) / last_row["SMA_long"]

def generate_signal(last_row, position, momentum_threshold=0.005):
    signal = None
    if last_row["close"] > last_row["SMA_long"] and last_row["momentum"] > momentum_threshold:
        if position == 0:
            signal = "BUY"
    elif last_row["close"] < last_row["SMA_long"]:
        if position == 1:
            signal = "SELL"
    return signal
