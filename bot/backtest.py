import time
import numpy as np
import pandas as pd
import yfinance as yf


def _flatten_columns(df):
    """Flatten MultiIndex columns returned by newer yfinance versions."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def _generate_synthetic_data(symbol, periods=500):
    """Generate synthetic OHLCV data for CI/testing when live download fails."""
    np.random.seed(abs(hash(symbol)) % (2**32))
    dates = pd.date_range(start="2025-01-01", periods=periods, freq="h")
    close = 400.0 * np.exp(np.cumsum(np.random.normal(0, 0.005, periods)))
    high = close * (1 + np.abs(np.random.normal(0, 0.003, periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.003, periods)))
    open_ = low + (high - low) * np.random.uniform(0, 1, periods)
    volume = np.random.randint(1_000_000, 10_000_000, periods).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def download_data(symbol):
    # Try yf.download first
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                start="2025-01-01",
                end="2025-12-31",
                interval="1h",
                progress=False,
                auto_adjust=True,
            )
            df = _flatten_columns(df)
            if df is not None and not df.empty:
                print(f"[{symbol}] Downloaded {len(df)} bars from yfinance (yf.download)")
                return df
        except Exception as e:
            print(f"Download failed for {symbol}, retry {attempt+1}: {e}")
        time.sleep(2)

    # Fallback: try yf.Ticker().history()
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start="2025-01-01", end="2025-12-31", interval="1h", auto_adjust=True)
        df = _flatten_columns(df)
        if df is not None and not df.empty:
            print(f"[{symbol}] Downloaded {len(df)} bars from yfinance (Ticker.history)")
            return df
    except Exception as e:
        print(f"Ticker history fallback failed for {symbol}: {e}")

    # Final fallback: synthetic data (for CI / offline environments)
    print(f"[{symbol}] WARNING: Using synthetic data (yfinance unavailable)")
    return _generate_synthetic_data(symbol)

def run_backtest():
    """
    Runs backtest for all symbols using config settings.
    Prints performance metrics and plots equity curve.
    """
    import matplotlib
    matplotlib.use("Agg")
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
    momentum_threshold = cfg.get("momentum_threshold", 0.005)
    use_trailing_stop = cfg.get("use_trailing_stop", False)
    trailing_stop_pct = cfg.get("trailing_stop_pct", 0.01)
    max_daily_loss = cfg.get("max_daily_loss", None)

    portfolio = {"equity_curve": [], "trades": []}
    # positions: {symbol: {"qty": int, "entry_price": float, "peak_price": float}}
    positions = {}

    # Fetch historical data
    hist_data = {}

    for symbol in symbols:
        df = download_data(symbol)

        # normalize column names
        df.columns = [c.lower() for c in df.columns]

        df = compute_indicators(
            df,
            cfg["sma_short"],
            cfg["sma_long"],
            cfg["momentum_period"]
        )

        hist_data[symbol] = df

    timestamps = hist_data[symbols[0]].index
    current_day = None
    day_start_equity = equity

    for t in timestamps:
        # Track daily starting equity for max daily loss check
        t_date = pd.Timestamp(t).date()
        if t_date != current_day:
            current_day = t_date
            day_start_equity = equity

        # Update equity curve snapshot (always, even when trading is halted)
        def _equity_snapshot():
            snap = equity
            for s, p in positions.items():
                if p["qty"] > 0:
                    snap += p["qty"] * hist_data[s].loc[t]["close"] - p["qty"] * p["entry_price"]
            return snap

        # Halt trading for rest of day if max daily loss is breached
        if max_daily_loss and day_start_equity > 0 and (day_start_equity - equity) / day_start_equity >= max_daily_loss:
            portfolio["equity_curve"].append({"timestamp": t, "equity": _equity_snapshot()})
            continue

        # Compute trend scores; skip timestamps where indicators are not yet ready (NaN)
        best_score = None
        best_symbol = None
        last_row = None
        for symbol in symbols:
            row = hist_data[symbol].loc[t]
            if pd.isna(row["SMA_long"]) or pd.isna(row["momentum"]):
                continue
            score = trend_score(row)
            if best_score is None or score > best_score:
                best_score = score
                best_symbol = symbol
                last_row = row

        if best_symbol is None or best_score <= 0:
            portfolio["equity_curve"].append({"timestamp": t, "equity": equity})
            continue

        # --- Trailing stop: check all open positions before generating new signal ---
        if use_trailing_stop:
            exits = []
            for symbol, p in positions.items():
                if p["qty"] == 0:
                    continue
                current_price = hist_data[symbol].loc[t]["close"]
                # Advance peak price upward
                if current_price > p["peak_price"]:
                    p["peak_price"] = current_price
                # Trigger trailing stop exit
                if current_price < p["peak_price"] * (1 - trailing_stop_pct):
                    exits.append((symbol, current_price))
            for symbol, exit_price in exits:
                p = positions[symbol]
                pnl = p["qty"] * (exit_price - p["entry_price"])
                equity += pnl
                portfolio["trades"].append({
                    "timestamp": t, "symbol": symbol, "action": "SELL",
                    "price": exit_price, "qty": p["qty"], "pnl": pnl,
                    "exit_reason": "trailing_stop",
                })
                positions[symbol] = {"qty": 0, "entry_price": 0, "peak_price": 0}

        # Check position and generate signal
        pos = positions.get(best_symbol, {"qty": 0, "entry_price": 0, "peak_price": 0})
        signal = generate_signal(last_row, 1 if pos["qty"] > 0 else 0,
                                 momentum_threshold=momentum_threshold)

        # Execute virtual trade
        if signal == "BUY" and pos["qty"] == 0:
            qty = int(equity * risk_per_trade / last_row["close"])
            if qty > 0:
                entry_price = last_row["close"]
                positions[best_symbol] = {"qty": qty, "entry_price": entry_price, "peak_price": entry_price}
                portfolio["trades"].append({
                    "timestamp": t, "symbol": best_symbol, "action": "BUY",
                    "price": entry_price, "qty": qty,
                })
        elif signal == "SELL" and pos["qty"] > 0:
            exit_price = last_row["close"]
            pnl = pos["qty"] * (exit_price - pos["entry_price"])
            equity += pnl
            portfolio["trades"].append({
                "timestamp": t, "symbol": best_symbol, "action": "SELL",
                "price": exit_price, "qty": pos["qty"], "pnl": pnl,
                "exit_reason": "signal",
            })
            positions[best_symbol] = {"qty": 0, "entry_price": 0, "peak_price": 0}

        portfolio["equity_curve"].append({"timestamp": t, "equity": _equity_snapshot()})

    # Close any remaining open positions at the last available price
    for symbol, p in list(positions.items()):
        if p["qty"] > 0:
            last_price = hist_data[symbol].iloc[-1]["close"]
            pnl = p["qty"] * (last_price - p["entry_price"])
            equity += pnl
            portfolio["trades"].append({
                "timestamp": timestamps[-1], "symbol": symbol, "action": "SELL",
                "price": last_price, "qty": p["qty"], "pnl": pnl,
                "exit_reason": "end_of_backtest",
            })
            positions[symbol] = {"qty": 0, "entry_price": 0, "peak_price": 0}

    eq_df = pd.DataFrame(portfolio["equity_curve"])
    eq_df.set_index("timestamp", inplace=True)

    # Metrics — handle case with no trades or no SELL trades
    trades_df = pd.DataFrame(portfolio["trades"]) if portfolio["trades"] else pd.DataFrame(columns=["action", "pnl"])
    if "pnl" in trades_df.columns and "action" in trades_df.columns:
        sell_trades = trades_df.loc[trades_df["action"] == "SELL"]
        sell_pnl = sell_trades["pnl"].dropna()
        wins = int((sell_pnl > 0).sum())
        total_sells = len(sell_pnl)
        total_buys = int((trades_df["action"] == "BUY").sum())
    else:
        wins = 0
        total_sells = 0
        total_buys = 0
    win_rate = wins / max(1, total_sells)
    max_drawdown = ((eq_df["equity"].cummax() - eq_df["equity"]) / eq_df["equity"].cummax()).max()

    print(f"\n--- Backtest Results ---")
    print(f"Final Equity: {equity:.2f}  (return: {(equity - initial_equity) / initial_equity * 100:.2f}%)")
    print(f"Trades: {total_buys} buys, {total_sells} sells ({min(total_buys, total_sells)} completed buy-sell pairs)")
    print(f"Win Rate: {win_rate*100:.2f}%  ({wins} winning / {total_sells} closed trades)")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")

    # Per-symbol breakdown
    if "symbol" in trades_df.columns and "action" in trades_df.columns:
        print("\n--- Per-Symbol Summary ---")
        for sym in symbols:
            sym_sells = trades_df.loc[(trades_df["symbol"] == sym) & (trades_df["action"] == "SELL")]
            sym_pnl = sym_sells["pnl"].dropna()
            sym_wins = int((sym_pnl > 0).sum())
            print(
                f"  {sym}: {len(sym_pnl)} closed trades | "
                f"P&L: {sym_pnl.sum():.2f} | "
                f"Win rate: {sym_wins / max(1, len(sym_pnl)) * 100:.0f}%"
            )

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(eq_df.index, eq_df["equity"], label="Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.savefig("backtest_equity_curve.png")
    plt.close()
