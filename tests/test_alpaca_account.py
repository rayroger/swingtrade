import os
"""
Unit tests for Alpaca account integration helpers.

These tests use mocks so they do not require live credentials or network access.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpaca.trading.client import TradingClient


_ACCOUNT = SimpleNamespace(
    account_number="PA1234567890",
    status="ACTIVE",
    equity="12345.67",
    cash="5000.00",
    buying_power="10000.00",
)

_POSITIONS = [
    SimpleNamespace(symbol="AAPL", qty="10", market_value="1750.00"),
    SimpleNamespace(symbol="TSLA", qty="5", market_value="900.00"),
]


@pytest.fixture
def trading_client():
    client = MagicMock(spec=TradingClient)
    client.get_account.return_value = _ACCOUNT
    client.get_all_positions.return_value = _POSITIONS
    return client


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


# ----------------------------
# Live (integration) tests
# ----------------------------

def _env_or_skip(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"Missing env var {name}; skipping live Alpaca tests")
    return value


@pytest.fixture(scope="session")
def live_trading_client():
    """
    Real Alpaca TradingClient constructed from env vars.
    Live tests run by default when secrets are present.
    """
    endpoint = _env_or_skip("ALPACA_ENDPOINT")
    key = _env_or_skip("ALPACA_KEY")
    secret = _env_or_skip("ALPACA_SECRET")

    # Your endpoint: https://paper-api.alpaca.markets/v2 -> paper=True
    paper = "paper-api" in endpoint.lower()

    return TradingClient(api_key=key, secret_key=secret, paper=paper)


@pytest.mark.integration
def test_live_account_identity(live_trading_client):
    account = live_trading_client.get_account()
    print(f"\n[LIVE] Account number : {account.account_number}")
    print(f"[LIVE] Status         : {account.status}")
    assert account.account_number, "account_number should not be empty"
    assert account.status, "status should not be empty"


@pytest.mark.integration
def test_live_account_balance(live_trading_client):
    account = live_trading_client.get_account()
    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)

    print(f"\n[LIVE] Equity        : ${equity:,.2f}")
    print(f"[LIVE] Cash          : ${cash:,.2f}")
    print(f"[LIVE] Buying power  : ${buying_power:,.2f}")

    assert equity >= 0, f"equity is negative: {equity}"
    assert cash >= 0, f"cash is negative: {cash}"
    assert buying_power >= 0, f"buying_power is negative: {buying_power}"


@pytest.mark.integration
def test_live_positions(live_trading_client):
    positions = live_trading_client.get_all_positions()
    assert positions is not None  # can be empty list

    print(f"\n[LIVE] Open positions ({len(positions)}):")
    for p in positions:
        print(f"  {p.symbol:6s}  qty={p.qty}  market_value=${float(p.market_value):,.2f}")
        assert p.symbol, "position symbol should not be empty"
        assert abs(float(p.qty)) > 1e-9, f"{p.symbol} qty should not be zero"

