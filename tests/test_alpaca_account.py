"""
Integration test: connect to Alpaca and verify account properties.

Requires environment variables:
  ALPACA_ENDPOINT  – base URL, e.g. https://paper-api.alpaca.markets
  ALPACA_KEY       – Alpaca API key ID
  ALPACA_SECRET    – Alpaca API secret key
"""
import os

import pytest
from alpaca.trading.client import TradingClient


@pytest.fixture(scope="module")
def trading_client():
    api_key = os.environ["ALPACA_KEY"]
    secret_key = os.environ["ALPACA_SECRET"]
    endpoint = os.environ.get("ALPACA_ENDPOINT", "")

    kwargs = dict(api_key=api_key, secret_key=secret_key)
    if endpoint:
        kwargs["url_override"] = endpoint

    return TradingClient(**kwargs)


def test_account_identity(trading_client):
    """Account number and status are present."""
    account = trading_client.get_account()
    print(f"\nAccount number : {account.account_number}")
    print(f"Status         : {account.status}")
    assert account.account_number, "account_number should not be empty"
    assert account.status, "status should not be empty"


def test_account_balance(trading_client):
    """Equity, cash, and buying power are non-negative numbers."""
    account = trading_client.get_account()
    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)

    print(f"\nEquity        : ${equity:,.2f}")
    print(f"Cash          : ${cash:,.2f}")
    print(f"Buying power  : ${buying_power:,.2f}")

    assert equity >= 0, f"equity is negative: {equity}"
    assert cash >= 0, f"cash is negative: {cash}"
    assert buying_power >= 0, f"buying_power is negative: {buying_power}"


def test_positions(trading_client):
    """Positions list is returned (may be empty); each entry has symbol and qty."""
    positions = trading_client.get_all_positions()
    print(f"\nOpen positions ({len(positions)}):")
    for p in positions:
        print(f"  {p.symbol:6s}  qty={p.qty}  market_value=${float(p.market_value):,.2f}")
        assert p.symbol, "position symbol should not be empty"
        assert abs(float(p.qty)) > 1e-9, f"{p.symbol} qty should not be zero"
