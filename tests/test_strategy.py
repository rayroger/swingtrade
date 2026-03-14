import pandas as pd
from bot.strategy import compute_indicators, generate_signal

def test_signal_output():
    data = pd.DataFrame({"close": [100 + i for i in range(60)]})
    df = compute_indicators(data)
    last = df.iloc[-1]
    signal = generate_signal(last, position=0)
    assert signal in ("BUY", "SELL", None)

