"""Strategy invariants: weights sum to 1, non-negative, and behave as designed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import strategies as st
from backtester.strategies import StrategyContext


def make_ctx(panel: pd.DataFrame, risk_free="CASH", date=None):
    return StrategyContext(
        columns=list(panel.columns),
        risk_free=risk_free,
        date=date or panel.index[-1],
        start_date=panel.index[0],
        prior_weights=None,
    )


def assert_valid_weights(w: pd.Series):
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w >= -1e-9).all()


def test_static_weights_normalize(synthetic_panel):
    ctx = make_ctx(synthetic_panel)
    w = st.static_weights(synthetic_panel, ctx, weights={"EQ": 3, "BOND": 1})  # unnormalized
    assert_valid_weights(w)
    assert w["EQ"] == pytest.approx(0.75)
    assert w["BOND"] == pytest.approx(0.25)


def test_equal_weight_excludes_cash(synthetic_panel):
    ctx = make_ctx(synthetic_panel)
    w = st.equal_weight(synthetic_panel, ctx, include_cash=False)
    assert_valid_weights(w)
    assert w["CASH"] == 0.0
    assert w["EQ"] == pytest.approx(1 / 3)


def test_inverse_vol_gives_low_vol_asset_more_weight(synthetic_panel):
    ctx = make_ctx(synthetic_panel)
    w = st.inverse_volatility(synthetic_panel, ctx, lookback=48, include_cash=False)
    assert_valid_weights(w)
    vols = synthetic_panel.drop(columns=["CASH"]).std(ddof=1)
    # The lowest-vol risky asset must carry the largest weight.
    assert w.drop("CASH").idxmax() == vols.idxmin()


def test_min_variance_beats_equal_weight_variance(synthetic_panel):
    ctx = make_ctx(synthetic_panel)
    risky = [c for c in synthetic_panel.columns if c != "CASH"]
    cov = synthetic_panel[risky].cov(ddof=1).values

    w_mv = st.min_variance(synthetic_panel, ctx, lookback=48, include_cash=False, shrinkage=0.0)
    assert_valid_weights(w_mv)
    var_mv = w_mv[risky].values @ cov @ w_mv[risky].values

    w_eq = np.full(len(risky), 1 / len(risky))
    var_eq = w_eq @ cov @ w_eq
    assert var_mv <= var_eq + 1e-12


def test_erc_equalizes_risk_contributions(synthetic_panel):
    ctx = make_ctx(synthetic_panel)
    risky = [c for c in synthetic_panel.columns if c != "CASH"]
    w = st.risk_parity_erc(synthetic_panel, ctx, lookback=48, include_cash=False, shrinkage=0.0)
    assert_valid_weights(w)

    cov = synthetic_panel[risky].cov(ddof=1).values
    wv = w[risky].values
    rc = wv * (cov @ wv)  # risk contributions
    # Contributions should be nearly equal: spread small vs their mean.
    assert (rc.max() - rc.min()) / rc.mean() < 0.05


def test_target_vol_parks_residual_in_cash(synthetic_panel):
    ctx = make_ctx(synthetic_panel)
    # A very low target forces heavy de-risking into cash.
    w = st.target_volatility(
        synthetic_panel, ctx, target_vol=0.01, lookback=48, base="equal_weight",
        base_params={"include_cash": False},
    )
    assert_valid_weights(w)
    assert w["CASH"] > 0.5  # most of the book sits in cash


def test_target_vol_never_levers(synthetic_panel):
    ctx = make_ctx(synthetic_panel)
    # A huge target can't create leverage: scaling k is capped at 1.
    w = st.target_volatility(
        synthetic_panel, ctx, target_vol=5.0, lookback=48, base="equal_weight",
        base_params={"include_cash": False},
    )
    assert_valid_weights(w)
    assert w["CASH"] == pytest.approx(0.0, abs=1e-9)


def test_glide_path_derisks_with_age(synthetic_panel):
    growth = ["EQ", "GOLD"]
    defensive = ["BOND", "CASH"]
    early = make_ctx(synthetic_panel, date=synthetic_panel.index[0])
    # 20 years later.
    late_date = synthetic_panel.index[0] + pd.DateOffset(years=20)
    late = StrategyContext(
        columns=list(synthetic_panel.columns), risk_free="CASH",
        date=late_date, start_date=synthetic_panel.index[0],
    )
    kw = dict(start_age=30, retirement_age=65, equity_start=0.9, equity_end=0.3,
              growth_assets=growth, defensive_assets=defensive)
    w_early = st.glide_path(synthetic_panel, early, **kw)
    w_late = st.glide_path(synthetic_panel, late, **kw)
    assert_valid_weights(w_early)
    assert_valid_weights(w_late)
    assert w_early[growth].sum() > w_late[growth].sum()  # de-risking over time
    assert w_early[growth].sum() == pytest.approx(0.9)
