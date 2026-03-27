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


def download_data(symbol, start="2022-01-01", end="2025-12-31"):
    # Try yf.download first
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
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
        df = ticker.history(start=start, end=end, interval="1h", auto_adjust=True)
        df = _flatten_columns(df)
        if df is not None and not df.empty:
            print(f"[{symbol}] Downloaded {len(df)} bars from yfinance (Ticker.history)")
            return df
    except Exception as e:
        print(f"Ticker history fallback failed for {symbol}: {e}")

    # Final fallback: synthetic data (for CI / offline environments)
    print(f"[{symbol}] WARNING: Using synthetic data (yfinance unavailable)")
    return _generate_synthetic_data(symbol)

def _indicators_ready(row):
    """Return True only when all required indicators have been computed (no NaNs)."""
    return not (pd.isna(row["SMA_long"]) or pd.isna(row["momentum"]) or pd.isna(row.get("volatility", float("nan"))))


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
    backtest_start = cfg.get("backtest_start", "2022-01-01")
    backtest_end = cfg.get("backtest_end", "2025-12-31")
    commission_per_trade = cfg.get("commission_per_trade", 0.0)
    slippage_pct = cfg.get("slippage_pct", 0.0)
    min_holding_bars = cfg.get("min_holding_bars", 0)

    portfolio = {"equity_curve": [], "trades": []}
    # positions: {symbol: {"qty": int, "entry_price": float, "peak_price": float, "entry_bar_idx": int}}
    positions = {}

    # Fetch historical data
    hist_data = {}

    for symbol in symbols:
        df = download_data(symbol, start=backtest_start, end=backtest_end)

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

    for bar_idx, t in enumerate(timestamps):
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
            if not _indicators_ready(row):
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
                exec_price = exit_price * (1 - slippage_pct)
                pnl = p["qty"] * (exec_price - p["entry_price"]) - commission_per_trade
                equity += pnl
                portfolio["trades"].append({
                    "timestamp": t, "symbol": symbol, "action": "SELL",
                    "price": exec_price, "qty": p["qty"], "pnl": pnl,
                    "exit_reason": "trailing_stop",
                })
                positions[symbol] = {"qty": 0, "entry_price": 0, "peak_price": 0, "entry_bar_idx": -1}

        # Check position and generate signal
        pos = positions.get(best_symbol, {"qty": 0, "entry_price": 0, "peak_price": 0, "entry_bar_idx": -1})
        signal = generate_signal(last_row, 1 if pos["qty"] > 0 else 0,
                                 momentum_threshold=momentum_threshold)

        # Suppress signal-based SELL until minimum holding period is satisfied
        # (trailing stop always fires regardless, handled above)
        if signal == "SELL" and pos["qty"] > 0:
            bars_held = bar_idx - pos.get("entry_bar_idx", bar_idx)
            if bars_held < min_holding_bars:
                signal = None

        # Execute virtual trade
        if signal == "BUY" and pos["qty"] == 0:
            exec_price = last_row["close"] * (1 + slippage_pct)
            qty = int(equity * risk_per_trade / exec_price)
            if qty > 0:
                equity -= commission_per_trade
                positions[best_symbol] = {
                    "qty": qty,
                    "entry_price": exec_price,
                    "peak_price": exec_price,
                    "entry_bar_idx": bar_idx,
                }
                portfolio["trades"].append({
                    "timestamp": t, "symbol": best_symbol, "action": "BUY",
                    "price": exec_price, "qty": qty,
                })
        elif signal == "SELL" and pos["qty"] > 0:
            exec_price = last_row["close"] * (1 - slippage_pct)
            pnl = pos["qty"] * (exec_price - pos["entry_price"]) - commission_per_trade
            equity += pnl
            portfolio["trades"].append({
                "timestamp": t, "symbol": best_symbol, "action": "SELL",
                "price": exec_price, "qty": pos["qty"], "pnl": pnl,
                "exit_reason": "signal",
            })
            positions[best_symbol] = {"qty": 0, "entry_price": 0, "peak_price": 0, "entry_bar_idx": -1}

        portfolio["equity_curve"].append({"timestamp": t, "equity": _equity_snapshot()})

    # Close any remaining open positions at the last available price
    for symbol, p in list(positions.items()):
        if p["qty"] > 0:
            last_close = hist_data[symbol].iloc[-1]["close"]
            exec_price = last_close * (1 - slippage_pct)
            pnl = p["qty"] * (exec_price - p["entry_price"]) - commission_per_trade
            equity += pnl
            portfolio["trades"].append({
                "timestamp": timestamps[-1], "symbol": symbol, "action": "SELL",
                "price": exec_price, "qty": p["qty"], "pnl": pnl,
                "exit_reason": "end_of_backtest",
            })
            positions[symbol] = {"qty": 0, "entry_price": 0, "peak_price": 0, "entry_bar_idx": -1}

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

    # Buy-and-hold SPY benchmark
    if "SPY" in hist_data:
        spy_close = hist_data["SPY"]["close"].dropna()
        if len(spy_close) >= 2:
            spy_bh_return = (spy_close.iloc[-1] - spy_close.iloc[0]) / spy_close.iloc[0]
            spy_bh_equity = initial_equity * (1 + spy_bh_return)
            print(f"Buy & Hold SPY:  {spy_bh_equity:.2f}  (return: {spy_bh_return * 100:.2f}%)")

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
    plt.plot(eq_df.index, eq_df["equity"], label="Strategy", color="steelblue")

    # Buy-and-hold SPY benchmark line
    if "SPY" in hist_data:
        spy_close = hist_data["SPY"]["close"].reindex(eq_df.index, method="ffill").dropna()
        if len(spy_close) >= 2:
            spy_bh_curve = initial_equity * spy_close / spy_close.iloc[0]
            plt.plot(spy_bh_curve.index, spy_bh_curve.values,
                     label="Buy & Hold SPY", linestyle="--", color="gray")

    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.savefig("backtest_equity_curve.png")
    plt.close()
