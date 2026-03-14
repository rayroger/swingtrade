def run_backtest():
    """
    Runs backtest for all symbols using config settings.
    Prints performance metrics and plots equity curve.
    """
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import yaml
    from bot.strategy import compute_indicators, trend_score, generate_signal

    # Load config
    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)

    symbols = cfg["symbols"]
    initial_equity = 10000
    equity = initial_equity
    risk_per_trade = cfg["risk_per_trade"]

    portfolio = {"equity_curve": [], "trades": []}
    positions = {}

    # Fetch historical data
    hist_data = {}
    for symbol in symbols:
        df = yf.download(symbol, start="2023-01-01", end="2023-12-31", interval="1h")
        df = compute_indicators(df, cfg["sma_short"], cfg["sma_long"], cfg["momentum_period"])
        hist_data[symbol] = df

    timestamps = hist_data[symbols[0]].index
    for t in timestamps:
        # Compute trend scores
        best_score = None
        best_symbol = None
        for symbol in symbols:
            row = hist_data[symbol].loc[t]
            score = trend_score(row)
            if best_score is None or score > best_score:
                best_score = score
                best_symbol = symbol
                last_row = row

        # Check position
        pos = positions.get(best_symbol, {"qty": 0, "entry_price": 0})
        signal = generate_signal(last_row, 1 if pos["qty"] > 0 else 0)

        # Execute virtual trade
        if signal == "BUY" and pos["qty"] == 0:
            qty = max(1, int(equity * risk_per_trade / last_row["Close"]))
            positions[best_symbol] = {"qty": qty, "entry_price": last_row["Close"]}
            portfolio["trades"].append({
                "timestamp": t, "symbol": best_symbol, "action": "BUY",
                "price": last_row["Close"], "qty": qty
            })
        elif signal == "SELL" and pos["qty"] > 0:
            exit_price = last_row["Close"]
            pnl = pos["qty"] * (exit_price - pos["entry_price"])
            equity += pnl
            portfolio["trades"].append({
                "timestamp": t, "symbol": best_symbol, "action": "SELL",
                "price": exit_price, "qty": pos["qty"], "pnl": pnl
            })
            positions[best_symbol] = {"qty": 0, "entry_price": 0}

        # Update equity curve
        equity_snapshot = equity
        for s, p in positions.items():
            equity_snapshot += p["qty"] * hist_data[s].loc[t]["Close"] - p["qty"] * p["entry_price"]
        portfolio["equity_curve"].append({"timestamp": t, "equity": equity_snapshot})

    eq_df = pd.DataFrame(portfolio["equity_curve"])
    eq_df.set_index("timestamp", inplace=True)

    trades_df = pd.DataFrame(portfolio["trades"])
    wins = trades_df[trades_df.get("pnl", 0) > 0].shape[0]
    losses = trades_df[trades_df.get("pnl", 0) <= 0].shape[0]
    win_rate = wins / max(1, wins + losses)
    max_drawdown = ((eq_df["equity"].cummax() - eq_df["equity"]) / eq_df["equity"].cummax()).max()

    print(f"Final Equity: {equity:.2f}")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")

    # Plot equity curve
    plt.figure(figsize=(12,6))
    plt.plot(eq_df.index, eq_df["equity"], label="Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.show()
