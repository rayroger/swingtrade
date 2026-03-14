import os
import pandas as pd
import alpaca_trade_api as tradeapi
import yaml

from strategy import compute_indicators, generate_signal, trend_score
from utils import log_trade

# Load config
with open("config/settings.yaml") as f:
    cfg = yaml.safe_load(f)

api = tradeapi.REST(
    key_id=os.environ["ALPACA_KEY"],
    secret_key=os.environ["ALPACA_SECRET"],
    base_url="https://paper-api.alpaca.markets"
)

account = api.get_account()
equity = float(account.equity)

best_score = None
best_symbol = None
positions = {p.symbol: int(p.qty) for p in api.list_positions()}

# Evaluate all symbols for trend score
for symbol in cfg['symbols']:
    bars = api.get_bars(symbol, tradeapi.TimeFrame.Hour, limit=100).df
    bars = compute_indicators(bars, cfg['sma_short'], cfg['sma_long'], cfg['momentum_period'])
    last = bars.iloc[-1]
    score = trend_score(last)
    if best_score is None or score > best_score:
        best_score = score
        best_symbol = symbol
        last_row = last
        current_position = 1 if symbol in positions else 0

signal = generate_signal(last_row, current_position)

if signal == "BUY":
    qty = max(1, int(equity * cfg['risk_per_trade'] / last_row['close']))
    api.submit_order(symbol=best_symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
    log_trade(best_symbol, "BUY", last_row['close'], qty, equity)

elif signal == "SELL" and best_symbol in positions:
    qty = positions[best_symbol]
    api.submit_order(symbol=best_symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
    log_trade(best_symbol, "SELL", last_row['close'], qty, equity)
