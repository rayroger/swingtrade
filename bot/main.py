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
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    import pandas as pd
    from bot.strategy import compute_indicators, generate_signal, trend_score
    from bot.utils import log_trade

    symbols = cfg["symbols"]
    alpaca_key = os.environ["ALPACA_KEY"]
    alpaca_secret = os.environ["ALPACA_SECRET"]

    # Configurable strategy params
    momentum_threshold = cfg.get("momentum_threshold", 0.005)


    # Connect to Alpaca using API key + secret
    trading_client = TradingClient(api_key=alpaca_key, secret_key=alpaca_secret, paper=True)
    data_client = StockHistoricalDataClient(api_key=alpaca_key, secret_key=alpaca_secret)

    account = trading_client.get_account()
    equity = float(account.equity)
    print(
        f"Connected to Alpaca | Account: {account.account_number} | Status: {account.status} | "
        f"Equity: ${equity:,.2f} | Cash: ${float(account.cash):,.2f} | "
        f"Buying power: ${float(account.buying_power):,.2f}"
    )

    positions = {p.symbol: int(float(p.qty)) for p in trading_client.get_all_positions()}  # qty is returned as a decimal string

    # Multi-symbol rotation: choose symbol with strongest trend
    best_score = None
    best_symbol = None
    best_last_row = None
    best_current_position = 0

    for symbol in symbols:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Hour,
            limit=100
        )
        bars = data_client.get_stock_bars(request_params).df

        # alpaca-py returns a MultiIndex (symbol, timestamp) DataFrame when symbol is a list
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")

        bars = compute_indicators(bars, cfg["sma_short"], cfg["sma_long"], cfg["momentum_period"])
        last = bars.iloc[-1]
        score = trend_score(last)

        if best_score is None or score > best_score:
            best_score = score
            best_symbol = symbol
            best_last_row = last
            best_current_position = 1 if symbol in positions else 0

    # Debug prints requested
    print(f"Best symbol + score: {best_symbol} | {best_score}")

    if best_last_row is not None:
        print(
            "Decision inputs | "
            f"close={float(best_last_row['close'])} | "
            f"SMA_long={float(best_last_row['SMA_long'])} | "
            f"momentum={float(best_last_row['momentum'])} | "
            f"momentum_threshold={float(momentum_threshold)} | "
            f"position={best_current_position}"
        )

    signal = generate_signal(last_row, current_position, momentum_threshold=momentum_threshold)
    print(f"Final signal: {signal}")

    if signal is None:
        print("No trade")

    # Execute trade
    if signal == "BUY":
        qty = max(1, int(equity * cfg["risk_per_trade"] / best_last_row["close"]))
        order = MarketOrderRequest(
            symbol=best_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order_data=order)
        log_trade(best_symbol, "BUY", best_last_row["close"], qty, equity)

    elif signal == "SELL" and best_symbol in positions:
        qty = positions[best_symbol]
        order = MarketOrderRequest(
            symbol=best_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC
        )
        trading_client.submit_order(order_data=order)
        log_trade(best_symbol, "SELL", best_last_row["close"], qty, equity)
