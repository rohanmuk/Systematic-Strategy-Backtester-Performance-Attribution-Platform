"""Allocation strategies: ``history -> target weights``.

Each strategy is a pure function that receives the return history available
**strictly up to and including the decision date** (the engine guarantees this,
so there is no lookahead) and returns a target weight vector over the full
universe that sums to 1. Strategies are registered by name so YAML configs can
select them via ``strategy.type``.

Estimation notes
----------------
* Covariance/volatility use a trailing window (``lookback`` months, default 36
  for covariance-based methods, 12 for inverse-vol). If less history is
  available early in the sample, whatever exists is used (min 2 obs).
* Covariance-based optimisers (min-variance, ERC) apply light shrinkage of the
  sample covariance toward its diagonal (``shrinkage`` δ, default 0.1):
  ``Σ̂ = (1−δ)Σ + δ·diag(Σ)``. This is a simple, transparent stand-in for
  Ledoit–Wolf that keeps the matrix well-conditioned and the optimiser stable
  without pulling in a heavier dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Coarse asset-class map, used by the glide path and for sensible defaults.
GROWTH_DEFAULT = ["VTI", "VEA", "VWO", "VNQ", "GLD"]
DEFENSIVE_DEFAULT = ["AGG", "TIP", "BIL"]


@dataclass
class StrategyContext:
    """Everything a strategy needs beyond the return history itself."""

    columns: List[str]  # full universe, defines output order
    risk_free: str  # cash proxy ticker
    date: pd.Timestamp  # decision date (weights apply to the *next* period)
    start_date: pd.Timestamp  # first month of the backtest (for the glide path)
    prior_weights: Optional[pd.Series] = None


StrategyFn = Callable[..., pd.Series]
STRATEGY_REGISTRY: Dict[str, StrategyFn] = {}


def register(name: str) -> Callable[[StrategyFn], StrategyFn]:
    def deco(fn: StrategyFn) -> StrategyFn:
        STRATEGY_REGISTRY[name] = fn
        return fn

    return deco


def get_strategy(name: str) -> StrategyFn:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _zeros(ctx: StrategyContext) -> pd.Series:
    return pd.Series(0.0, index=ctx.columns)


def _normalize(w: pd.Series) -> pd.Series:
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights sum to zero; cannot normalise.")
    return w / total


def _risky_assets(ctx: StrategyContext, include_cash: bool) -> List[str]:
    if include_cash:
        return list(ctx.columns)
    return [c for c in ctx.columns if c != ctx.risk_free]


def _trailing(history: pd.DataFrame, lookback: int, cols: List[str]) -> pd.DataFrame:
    window = history[cols].tail(lookback)
    return window.dropna(how="all")


def _shrink_cov(cov: np.ndarray, delta: float) -> np.ndarray:
    """Shrink a sample covariance toward its diagonal by fraction ``delta``."""
    return (1.0 - delta) * cov + delta * np.diag(np.diag(cov))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@register("static_weights")
def static_weights(history: pd.DataFrame, ctx: StrategyContext, *, weights: Dict[str, float]) -> pd.Series:
    """Fixed target weights (e.g. 60/40). Renormalised to sum to 1."""
    w = _zeros(ctx)
    for ticker, weight in weights.items():
        if ticker not in w.index:
            raise KeyError(f"static_weights references {ticker!r} not in universe {ctx.columns}")
        w[ticker] = float(weight)
    return _normalize(w)


@register("equal_weight")
def equal_weight(history: pd.DataFrame, ctx: StrategyContext, *, include_cash: bool = True) -> pd.Series:
    """Equal weight (1/N) across the universe (optionally excluding cash)."""
    assets = _risky_assets(ctx, include_cash)
    w = _zeros(ctx)
    w[assets] = 1.0 / len(assets)
    return w


@register("inverse_volatility")
def inverse_volatility(
    history: pd.DataFrame,
    ctx: StrategyContext,
    *,
    lookback: int = 12,
    include_cash: bool = False,
) -> pd.Series:
    """Inverse-volatility (naive risk parity): ``w_i ∝ 1/σ_i``.

    Volatilities from the trailing ``lookback`` window. Cash is excluded by
    default (its ~zero vol would otherwise absorb almost all the weight).
    """
    assets = _risky_assets(ctx, include_cash)
    window = _trailing(history, lookback, assets)
    vol = window.std(ddof=1)
    vol = vol.replace(0.0, np.nan).dropna()
    if vol.empty:
        return equal_weight(history, ctx, include_cash=include_cash)
    inv = 1.0 / vol
    w = _zeros(ctx)
    w[inv.index] = inv
    return _normalize(w)


@register("min_variance")
def min_variance(
    history: pd.DataFrame,
    ctx: StrategyContext,
    *,
    lookback: int = 36,
    include_cash: bool = False,
    shrinkage: float = 0.1,
) -> pd.Series:
    """Long-only global minimum-variance portfolio.

    Solves ``min_w wᵀΣ̂w`` s.t. ``Σw = 1, w ≥ 0`` with SLSQP, where ``Σ̂`` is the
    diagonally-shrunk sample covariance over the trailing window.
    """
    assets = _risky_assets(ctx, include_cash)
    window = _trailing(history, lookback, assets).dropna(axis=1, how="any")
    assets = list(window.columns)
    if len(assets) < 2:
        return equal_weight(history, ctx, include_cash=include_cash)

    cov = _shrink_cov(window.cov(ddof=1).values, shrinkage)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    w = _solve_long_only(objective, len(assets))
    out = _zeros(ctx)
    out[assets] = w
    return _normalize(out)


@register("risk_parity")
def risk_parity_erc(
    history: pd.DataFrame,
    ctx: StrategyContext,
    *,
    lookback: int = 36,
    include_cash: bool = False,
    shrinkage: float = 0.1,
) -> pd.Series:
    """Full equal-risk-contribution (ERC) portfolio.

    Each asset contributes an equal share of total portfolio variance. Risk
    contribution ``RC_i = w_i · (Σw)_i``; total risk ``= Σ_i RC_i = wᵀΣw``. We
    minimise the dispersion of risk contributions
    ``Σ_i Σ_j (RC_i − RC_j)²`` s.t. ``Σw = 1, w ≥ 0``. Unlike inverse-vol, ERC
    accounts for correlations, so correlated assets are down-weighted.
    """
    assets = _risky_assets(ctx, include_cash)
    window = _trailing(history, lookback, assets).dropna(axis=1, how="any")
    assets = list(window.columns)
    if len(assets) < 2:
        return equal_weight(history, ctx, include_cash=include_cash)

    cov = _shrink_cov(window.cov(ddof=1).values, shrinkage)

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        marginal = cov @ w
        rc = w * marginal  # risk contribution of each asset
        # Sum of squared pairwise differences == 2n·Var(rc); minimise dispersion.
        return float(np.sum((rc[:, None] - rc[None, :]) ** 2) / (port_var + 1e-12))

    w = _solve_long_only(objective, len(assets), equalish_start=True)
    out = _zeros(ctx)
    out[assets] = w
    return _normalize(out)


@register("target_volatility")
def target_volatility(
    history: pd.DataFrame,
    ctx: StrategyContext,
    *,
    target_vol: float = 0.10,
    lookback: int = 12,
    base: str = "equal_weight",
    base_params: Optional[Dict] = None,
) -> pd.Series:
    """Volatility-targeting overlay on a base risky portfolio.

    Computes base (risky) weights, estimates the base portfolio's annualised
    volatility over the trailing window, then scales exposure by
    ``k = min(1, target_vol / base_vol)`` — the residual ``1−k`` is parked in
    cash. We deliberately **cap k at 1 (no leverage)**, so the strategy can
    de-risk but never borrows. ``target_vol`` is annualised (e.g. 0.10 = 10%).
    """
    from . import MONTHS_PER_YEAR

    base_fn = get_strategy(base)
    base_params = base_params or {}
    base_w = base_fn(history, ctx, **base_params)

    # Realised vol of the base portfolio over the trailing window.
    window = history[ctx.columns].tail(lookback).fillna(0.0)
    port_ret = window.mul(base_w, axis=1).sum(axis=1)
    base_vol = float(port_ret.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))

    if base_vol <= 0:
        return base_w
    k = min(1.0, target_vol / base_vol)

    w = base_w * k
    w[ctx.risk_free] = w.get(ctx.risk_free, 0.0) + (1.0 - k)
    return _normalize(w)


@register("glide_path")
def glide_path(
    history: pd.DataFrame,
    ctx: StrategyContext,
    *,
    start_age: int = 30,
    retirement_age: int = 65,
    equity_start: float = 0.90,
    equity_end: float = 0.30,
    growth_assets: Optional[List[str]] = None,
    defensive_assets: Optional[List[str]] = None,
) -> pd.Series:
    """Age-based glide path: de-risk linearly with age.

    The growth (equity-like) share glides linearly from ``equity_start`` at
    ``start_age`` to ``equity_end`` at ``retirement_age``, then stays flat. Age
    advances with calendar time since the backtest start. The growth share is
    split equally across available growth assets; the remainder equally across
    defensive assets. This is time-driven (not data-driven) by construction.
    """
    growth = [t for t in (growth_assets or GROWTH_DEFAULT) if t in ctx.columns]
    defensive = [t for t in (defensive_assets or DEFENSIVE_DEFAULT) if t in ctx.columns]
    if not growth or not defensive:
        raise ValueError("glide_path needs at least one growth and one defensive asset in the universe")

    years_elapsed = (ctx.date - ctx.start_date).days / 365.25
    age = start_age + years_elapsed
    span = max(retirement_age - start_age, 1)
    frac = (age - start_age) / span
    frac = min(max(frac, 0.0), 1.0)
    equity_share = equity_start + frac * (equity_end - equity_start)

    w = _zeros(ctx)
    w[growth] = equity_share / len(growth)
    w[defensive] = (1.0 - equity_share) / len(defensive)
    return _normalize(w)


# ---------------------------------------------------------------------------
# Optimiser plumbing
# ---------------------------------------------------------------------------

def _solve_long_only(objective, n: int, equalish_start: bool = False) -> np.ndarray:
    """Minimise ``objective(w)`` over the long-only simplex (Σw=1, w≥0)."""
    x0 = np.full(n, 1.0 / n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        logger.warning("Optimiser did not converge (%s); falling back to equal weight.", res.message)
        return x0
    # Clean tiny negatives from numerical noise.
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()
