import os
import yaml

# Load config
with open("config/settings.yaml") as f:
    cfg = yaml.safe_load(f)

# Mode priority:
# 1) GitHub workflow parameter
# 2) settings.yaml
# 3) default = live

mode = os.environ.get("BOT_MODE") or cfg.get("mode", "live")

print(f"Running bot in mode: {mode}")

if mode == "backtest":
    from bot.backtest import run_backtest
    run_backtest()
else:
    import pandas as pd
    import alpaca_trade_api as tradeapi
    from bot.strategy import compute_indicators, generate_signal, trend_score
    from bot.utils import log_trade

    # Load config
    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)

    symbols = cfg["symbols"]

    # Connect to Alpaca
    api = tradeapi.REST(
        key_id=os.environ["ALPACA_KEY"],
        secret_key=os.environ["ALPACA_SECRET"],
        base_url="https://paper-api.alpaca.markets"
    )

    account = api.get_account()
    equity = float(account.equity)

    positions = {p.symbol: int(p.qty) for p in api.list_positions()}

    # Multi-symbol rotation: choose symbol with strongest trend
    best_score = None
    best_symbol = None
    for symbol in symbols:
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Hour, limit=100).df
        bars = compute_indicators(bars, cfg["sma_short"], cfg["sma_long"], cfg["momentum_period"])
        last = bars.iloc[-1]
        score = trend_score(last)
        if best_score is None or score > best_score:
            best_score = score
            best_symbol = symbol
            last_row = last
            current_position = 1 if symbol in positions else 0

    signal = generate_signal(last_row, current_position)

    # Execute trade
    if signal == "BUY":
        qty = max(1, int(equity * cfg["risk_per_trade"] / last_row["close"]))
        api.submit_order(symbol=best_symbol, qty=qty, side="buy", type="market", time_in_force="gtc")
        log_trade(best_symbol, "BUY", last_row["close"], qty, equity)

    elif signal == "SELL" and best_symbol in positions:
        qty = positions[best_symbol]
        api.submit_order(symbol=best_symbol, qty=qty, side="sell", type="market", time_in_force="gtc")
        log_trade(best_symbol, "SELL", last_row["close"], qty, equity)
