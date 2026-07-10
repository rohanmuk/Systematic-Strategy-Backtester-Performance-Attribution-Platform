"""Metrics validated against hand-computed values and cross-checked vs empyrical.

The hand-computed cases are the primary evidence (they pin the exact formula);
the empyrical cross-checks are secondary and skipped if the library is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import metrics


# ---------------------------------------------------------------------------
# Hand-computed cases
# ---------------------------------------------------------------------------

def test_cagr_hand_computed(known_returns):
    # CAGR = (∏(1+r))^(12/n) - 1. Independently compute the product, then also
    # pin the literal value (≈ 12.37%).
    expected = float(np.prod(1.0 + known_returns.values)) ** (12 / len(known_returns)) - 1
    assert metrics.cagr(known_returns) == pytest.approx(expected, rel=1e-12)
    assert metrics.cagr(known_returns) == pytest.approx(0.123735, abs=1e-6)


def test_annualized_vol_hand_computed(known_returns):
    # sample std of the 6 returns * sqrt(12)
    expected = known_returns.std(ddof=1) * np.sqrt(12)
    assert metrics.annualized_volatility(known_returns) == pytest.approx(expected)
    # pinned numeric value
    assert metrics.annualized_volatility(known_returns) == pytest.approx(0.081972, abs=1e-5)


def test_sharpe_hand_computed(known_returns):
    # mean=0.01, std(ddof=1)=0.0236643 -> Sharpe = 0.01/0.0236643*sqrt(12)
    assert metrics.sharpe_ratio(known_returns, rf=0.0) == pytest.approx(1.46385, abs=1e-4)


def test_zero_vol_series_is_safe():
    r = pd.Series([0.01] * 12, index=pd.date_range("2020-01-31", periods=12, freq="ME"))
    assert metrics.annualized_volatility(r) == pytest.approx(0.0)
    assert np.isnan(metrics.sharpe_ratio(r))  # zero denominator -> NaN, not a crash
    assert metrics.cagr(r) == pytest.approx((1.01) ** 12 - 1)


def test_max_drawdown_hand_computed():
    # wealth: 1.10 then 0.99 -> trough drawdown = 0.99/1.10 - 1 = -0.10
    r = pd.Series([0.10, -0.10], index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    assert metrics.max_drawdown(r) == pytest.approx(0.10)


def test_calmar_is_cagr_over_maxdd():
    r = pd.Series([0.02, -0.03, 0.04, -0.01, 0.02, 0.01],
                  index=pd.date_range("2020-01-31", periods=6, freq="ME"))
    assert metrics.calmar_ratio(r) == pytest.approx(metrics.cagr(r) / metrics.max_drawdown(r))


def test_sortino_only_penalizes_downside():
    # All-positive returns -> zero downside deviation -> NaN Sortino.
    up = pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    assert np.isnan(metrics.sortino_ratio(up))


def test_downside_deviation_hand_computed():
    r = pd.Series([0.02, -0.02, -0.04, 0.06], index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    # min(r,0)^2 = [0, 4e-4, 16e-4, 0]; mean over 4 = 5e-4; sqrt * sqrt(12)
    expected = np.sqrt(5e-4) * np.sqrt(12)
    assert metrics.downside_deviation(r) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------

def test_beta_of_scaled_benchmark_is_scale():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.005, 0.03, 24), index=idx)
    strat = 2.0 * bench  # exactly 2x exposure, no idiosyncratic noise
    assert metrics.beta(strat, bench) == pytest.approx(2.0)


def test_alpha_zero_when_capm_holds():
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.005, 0.03, 36), index=idx)
    rf = 0.001
    strat = rf + 1.3 * (bench - rf)  # CAPM-exact -> alpha 0
    assert metrics.alpha(strat, bench, rf=rf) == pytest.approx(0.0, abs=1e-9)
    assert metrics.beta(strat, bench) == pytest.approx(1.3)


def test_tracking_error_zero_for_constant_active():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0.005, 0.03, 24), index=idx)
    strat = bench + 0.002  # constant active return
    assert metrics.tracking_error(strat, bench) == pytest.approx(0.0, abs=1e-12)
    assert metrics.active_return(strat, bench) == pytest.approx(0.002 * 12)


def test_capture_is_one_when_identical():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.005, 0.03, 24), index=idx)
    assert metrics.up_capture(bench, bench) == pytest.approx(1.0)
    assert metrics.down_capture(bench, bench) == pytest.approx(1.0)


def test_information_ratio_sign_matches_outperformance():
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.005, 0.03, 36), index=idx)
    strat = bench + rng.normal(0.003, 0.005, 36)  # positive average active
    assert metrics.information_ratio(strat, bench) > 0


# ---------------------------------------------------------------------------
# Cross-checks against empyrical (secondary; skipped if unavailable)
# ---------------------------------------------------------------------------

empyrical = pytest.importorskip("empyrical")


def test_xcheck_max_drawdown_vs_empyrical(known_returns):
    ours = -metrics.max_drawdown(known_returns)  # empyrical returns a negative number
    assert ours == pytest.approx(empyrical.max_drawdown(known_returns), abs=1e-9)


def test_xcheck_cagr_vs_empyrical():
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.006, 0.03, 60), index=idx)
    assert metrics.cagr(r) == pytest.approx(
        empyrical.annual_return(r, period="monthly"), rel=1e-6
    )


def test_xcheck_annual_vol_vs_empyrical():
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.006, 0.03, 60), index=idx)
    assert metrics.annualized_volatility(r) == pytest.approx(
        empyrical.annual_volatility(r, period="monthly"), rel=1e-6
    )
