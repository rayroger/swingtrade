import csv
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("logs/trades.csv")

def log_trade(symbol, action, price, qty, equity):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, symbol, action, price, qty, equity])
