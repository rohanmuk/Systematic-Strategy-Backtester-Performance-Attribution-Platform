"""Per-asset attribution: contribution to return and contribution to risk.

These decompositions answer "where did the return / risk come from?" — the
questions an allocator is asked in a review.

Contribution to return
    For each period the portfolio's (gross) return is ``Σ_i w_{i,t} r_{i,t}``
    where ``w_{i,t}`` is the weight held at the start of period ``t``. Summing a
    single asset's ``w_{i,t} r_{i,t}`` over all periods gives its total
    contribution; the asset contributions sum to the **arithmetic** sum of
    portfolio returns (not the compounded total — compounding is path-dependent
    and cannot be cleanly split by asset).

Contribution to risk
    Using average weights ``w̄`` and the sample covariance ``Σ`` of asset
    returns over the window, total volatility ``σ_p = sqrt(w̄ᵀ Σ w̄)`` decomposes
    exactly as ``σ_p = Σ_i RC_i`` with risk contribution
    ``RC_i = w̄_i · (Σ w̄)_i / σ_p``. Percent contributions sum to 100%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import MONTHS_PER_YEAR


@dataclass
class Attribution:
    to_return: pd.DataFrame  # per-asset total & % contribution to return
    to_risk: pd.DataFrame  # per-asset weight, MCR, RC, % contribution to risk


def contribution_to_return(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Total (arithmetic) return contributed by each asset over the backtest."""
    aligned = asset_returns.reindex(index=weights.index, columns=weights.columns)
    per_period = weights * aligned
    total = per_period.sum(axis=0)
    total_sum = total.sum()
    pct = total / total_sum if total_sum != 0 else total * np.nan
    return pd.DataFrame(
        {"contribution": total, "pct_of_total": pct}
    ).sort_values("contribution", ascending=False)


def contribution_to_risk(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Risk decomposition using average weights and the sample covariance.

    Returns a per-asset table of average weight, marginal risk contribution
    (annualised), risk contribution (annualised), and percent of total risk.
    """
    assets = list(weights.columns)
    aligned = asset_returns.reindex(index=weights.index, columns=assets)

    w_bar = weights.mean(axis=0).values
    cov = aligned.cov(ddof=1).values  # monthly covariance
    port_var = float(w_bar @ cov @ w_bar)
    sigma_p = float(np.sqrt(port_var)) if port_var > 0 else np.nan

    marginal = cov @ w_bar  # ∂σ²/∂w scaled below
    if sigma_p and not np.isnan(sigma_p):
        mcr = marginal / sigma_p  # marginal contribution to σ_p
        rc = w_bar * mcr  # risk contribution; Σ rc == σ_p
        pct = rc / sigma_p
    else:  # pragma: no cover - degenerate zero-variance window
        mcr = rc = pct = np.full(len(assets), np.nan)

    ann = np.sqrt(MONTHS_PER_YEAR)
    return pd.DataFrame(
        {
            "avg_weight": w_bar,
            "marginal_risk_contribution": mcr * ann,
            "risk_contribution": rc * ann,
            "pct_of_risk": pct,
        },
        index=assets,
    ).sort_values("pct_of_risk", ascending=False)


def compute_attribution(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> Attribution:
    return Attribution(
        to_return=contribution_to_return(weights, asset_returns),
        to_risk=contribution_to_risk(weights, asset_returns),
    )


def turnover_summary(turnover: pd.Series, costs: pd.Series) -> dict:
    """Headline turnover and cost-drag figures.

    ``annual_turnover`` is average one-way turnover per year; ``annual_cost_drag``
    is the average annual return lost to transaction costs.
    """
    n_years = len(turnover) / MONTHS_PER_YEAR if len(turnover) else np.nan
    total_turnover = float(turnover.sum())
    total_cost = float(costs.sum())
    return {
        "total_one_way_turnover": total_turnover,
        "annual_turnover": total_turnover / n_years if n_years else np.nan,
        "total_cost_drag": total_cost,
        "annual_cost_drag": total_cost / n_years if n_years else np.nan,
    }
