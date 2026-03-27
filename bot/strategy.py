import pandas as pd

def compute_indicators(df, sma_short=10, sma_long=50, momentum_period=10):
    df["SMA_short"] = df["close"].rolling(sma_short).mean()
    df["SMA_long"] = df["close"].rolling(sma_long).mean()
    df["momentum"] = df["close"].pct_change(momentum_period)
    # Rolling volatility (std of 1-period returns) normalizes trend_score across
    # symbols with different inherent volatility (e.g. TQQQ vs SPY/QQQ).
    df["volatility"] = df["close"].pct_change(1).rolling(momentum_period).std()
    return df

def trend_score(last_row):
    # Normalize by rolling volatility so that a 3× leveraged ETF does not
    # automatically dominate the multi-symbol rotation just because its raw
    # momentum magnitude is ~3× larger.
    vol = last_row["volatility"]
    if pd.isna(vol) or vol <= 0:
        return 0.0
    return (last_row["momentum"] / vol) * (last_row["SMA_short"] - last_row["SMA_long"]) / last_row["SMA_long"]

def generate_signal(last_row, position, momentum_threshold=0.005):
    signal = None
    if last_row["close"] > last_row["SMA_long"] and last_row["momentum"] > momentum_threshold:
        if position == 0:
            signal = "BUY"
    elif last_row["close"] < last_row["SMA_long"]:
        if position == 1:
            signal = "SELL"
    return signal
