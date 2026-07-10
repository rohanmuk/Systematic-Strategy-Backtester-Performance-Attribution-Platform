"""Engine invariants: accounting identity, capital conservation, no-lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.config import (
    BacktestConfig,
    RebalanceConfig,
    StrategyConfig,
)
from backtester.engine import run_backtest


def _panel(seed=0, n=60):
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "BIL": rng.normal(0.001, 0.0005, n),
            "AGG": rng.normal(0.003, 0.012, n),
            "VTI": rng.normal(0.007, 0.045, n),
            "GLD": rng.normal(0.004, 0.045, n),
        },
        index=idx,
    )


def _cfg(strategy_type="equal_weight", params=None, method="calendar", freq="quarterly", cost_bps=10.0):
    return BacktestConfig(
        name="test",
        tickers=["BIL", "AGG", "VTI", "GLD"],
        risk_free="BIL",
        strategy=StrategyConfig(type=strategy_type, params=params or {}),
        rebalance=RebalanceConfig(method=method, frequency=freq, cost_bps=cost_bps),
    )


def test_weights_sum_to_one_every_period():
    res = run_backtest(_panel(), _cfg("min_variance", {"lookback": 24}), warmup=12)
    row_sums = res.weights.sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-6)


def test_accounting_identity_gross_return():
    """Gross period return must equal held weights · asset returns (no leakage)."""
    panel = _panel()
    res = run_backtest(panel, _cfg("equal_weight", {"include_cash": True}), warmup=12)
    for date in res.returns.index:
        expected = float((res.weights.loc[date] * panel.loc[date]).sum())
        assert res.gross_returns.loc[date] == pytest.approx(expected, abs=1e-12)


def test_costs_reduce_net_returns():
    panel = _panel()
    free = run_backtest(panel, _cfg("inverse_volatility", {"lookback": 12}, freq="monthly", cost_bps=0.0), warmup=12)
    costly = run_backtest(panel, _cfg("inverse_volatility", {"lookback": 12}, freq="monthly", cost_bps=50.0), warmup=12)
    # Same gross path, but costs drag net wealth strictly lower.
    assert np.allclose(free.gross_returns.values, costly.gross_returns.values, atol=1e-12)
    assert costly.wealth.iloc[-1] < free.wealth.iloc[-1]
    assert costly.costs.sum() > 0


def test_buy_and_hold_never_trades_but_drifts():
    panel = _panel()
    res = run_backtest(
        panel, _cfg("static_weights", {"weights": {"VTI": 0.6, "AGG": 0.4}}, method="buy_and_hold"),
        warmup=12,
    )
    assert res.turnover.sum() == pytest.approx(0.0)
    assert res.costs.sum() == pytest.approx(0.0)
    # Weights must drift (not be constant) since we never rebalance.
    assert res.weights["VTI"].std() > 0


def test_no_lookahead():
    """Perturbing only the final month must not change any earlier weight/return.

    Weights at month t are formed from data through t and earn month t+1, so a
    change to the last observation cannot propagate backwards.
    """
    panel = _panel(seed=5)
    cfg = _cfg("min_variance", {"lookback": 24}, freq="monthly")
    base = run_backtest(panel, cfg, warmup=12)

    perturbed_panel = panel.copy()
    perturbed_panel.iloc[-1] = perturbed_panel.iloc[-1] * 5.0 + 0.05  # shock the last month
    perturbed = run_backtest(perturbed_panel, cfg, warmup=12)

    # All weights identical (last period's weights were decided before the shock).
    pd.testing.assert_frame_equal(base.weights, perturbed.weights)
    # All returns identical except the final month, which earns the shocked data.
    pd.testing.assert_series_equal(base.returns.iloc[:-1], perturbed.returns.iloc[:-1])
    assert base.returns.iloc[-1] != perturbed.returns.iloc[-1]


def test_capital_conservation_reconstructs_wealth():
    """Net wealth equals the compounded net return path from initial capital."""
    panel = _panel()
    res = run_backtest(panel, _cfg("equal_weight"), warmup=12)
    reconstructed = res.initial_capital * (1.0 + res.returns).cumprod()
    assert np.allclose(reconstructed.values, res.initial_capital * res.wealth.values)
