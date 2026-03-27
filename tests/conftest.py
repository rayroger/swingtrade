import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as live integration tests requiring real credentials",
    )
