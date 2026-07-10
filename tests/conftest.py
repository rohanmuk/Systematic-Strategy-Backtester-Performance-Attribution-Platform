"""Shared fixtures: small synthetic panels with known properties (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def monthly_index():
    return pd.date_range("2015-01-31", periods=48, freq="ME")


@pytest.fixture
def known_returns():
    """A tiny return series with hand-computable statistics (see test_metrics)."""
    return pd.Series(
        [0.03, -0.01, 0.02, 0.00, 0.04, -0.02],
        index=pd.date_range("2020-01-31", periods=6, freq="ME"),
        name="strat",
    )


@pytest.fixture
def synthetic_panel(monthly_index):
    """A reproducible 4-asset monthly return panel with distinct vols/correlations."""
    rng = np.random.default_rng(42)
    n = len(monthly_index)
    # Asset vols (monthly): low -> high.
    market = rng.normal(0.006, 0.04, n)
    data = {
        "CASH": rng.normal(0.001, 0.0005, n),
        "BOND": 0.3 * market + rng.normal(0.002, 0.010, n),
        "EQ": market + rng.normal(0.003, 0.015, n),
        "GOLD": rng.normal(0.004, 0.045, n),
    }
    return pd.DataFrame(data, index=monthly_index)
