import pandas as pd
import yfinance as yf
from strategy import compute_indicators, generate_signal

def backtest(symbol, start_date, end_date, initial_equity=10000):
    df = yf.download(symbol, start=start_date, end=end_date, interval="1h")
    df = compute_indicators(df)
    equity = initial_equity
    position = 0
    trades = []

    for i, row in df.iterrows():
        signal = generate_signal(row, position)
        if signal == "BUY":
            qty = max(1, int(equity * 0.02 / row['Close']))
            entry_price = row['Close']
            position = qty
            trades.append({"timestamp": i, "symbol": symbol, "action": "BUY", "price": entry_price, "qty": qty})
        elif signal == "SELL" and position > 0:
            exit_price = row['Close']
            equity += position * (exit_price - entry_price)
            trades.append({"timestamp": i, "symbol": symbol, "action": "SELL", "price": exit_price, "qty": position})
            position = 0

    return trades, equity
