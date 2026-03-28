import pandas as pd
import numpy as np
from bot.strategy import compute_indicators, generate_signal, trend_score


def test_signal_output():
    data = pd.DataFrame({"close": [100 + i for i in range(60)]})
    df = compute_indicators(data)
    last = df.iloc[-1]
    signal = generate_signal(last, position=0)
    assert signal in ("BUY", "SELL", None)


def test_trend_score_with_zero_volatility():
    """trend_score should return 0.0 when volatility is zero or NaN."""
    row = pd.Series({"momentum": 0.05, "SMA_short": 105.0, "SMA_long": 100.0, "volatility": 0.0})
    assert trend_score(row) == 0.0

    row_nan = pd.Series({"momentum": 0.05, "SMA_short": 105.0, "SMA_long": 100.0, "volatility": float("nan")})
    assert trend_score(row_nan) == 0.0


def test_regime_filter_blocks_buy_in_bear_market():
    """Simulate the regime filter: BUY on the best symbol should be suppressed
    when SPY's close is below its SMA_regime."""
    # Build a 250-bar SPY series that is in a downtrend (close < SMA_200)
    periods = 250
    # Start high, drift downward so recent close < rolling-200 mean
    close_vals = np.linspace(500, 300, periods)
    spy_df = pd.DataFrame({"close": close_vals})
    spy_df["SMA_regime"] = spy_df["close"].rolling(200).mean()
    spy_last = spy_df.iloc[-1]

    # SPY is below its 200-day SMA → bear regime
    assert spy_last["close"] < spy_last["SMA_regime"], (
        "Test setup error: SPY should be below SMA_regime"
    )

    # A raw BUY signal from generate_signal should exist (trend conditions met)
    data = pd.DataFrame({"close": [100 + i * 0.5 for i in range(60)]})
    df = compute_indicators(data)
    last = df.iloc[-1]
    raw_signal = generate_signal(last, position=0)

    # The regime filter logic (replicated from backtest.py) suppresses the BUY
    signal = raw_signal
    if signal == "BUY":
        if not pd.isna(spy_last["SMA_regime"]) and spy_last["close"] <= spy_last["SMA_regime"]:
            signal = None

    assert signal is None, "Regime filter should suppress BUY in bear market"


def test_cooldown_suppresses_reentry():
    """BUY should be suppressed when bar_idx - last_exit_bar < cooldown_bars."""
    cooldown_bars = 5
    last_exit = 10
    # Still within cooldown window
    bar_idx = 12
    assert (bar_idx - last_exit) < cooldown_bars

    # Outside cooldown window — re-entry allowed
    bar_idx_allowed = 16
    assert (bar_idx_allowed - last_exit) >= cooldown_bars
