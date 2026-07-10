"""Rebalancing policies and trading-cost accounting.

Two policies are supported:

* **Calendar** — trade back to target every *N* months (monthly / quarterly /
  annual), regardless of drift.
* **Threshold (drift band)** — trade back to target only when some asset's
  actual (drifted) weight has moved more than ``band`` away from its target.

``buy_and_hold`` is the degenerate case: allocate once, never trade again.

Turnover & cost convention
--------------------------
For a rebalance from current weights ``w`` to target ``w*``:

* ``Δ = Σ_i |w*_i − w_i|`` is the total absolute weight change. Because both
  vectors sum to 1, buys and sells each equal ``Δ/2``.
* **One-way turnover** (the reported figure) is ``Δ/2``.
* **Cost drag** is ``(cost_bps / 10_000) · Δ`` — i.e. the per-side cost in bps
  applied to the notional traded on *each* leg (sells of ``Δ/2`` plus buys of
  ``Δ/2``). The initial allocation is assumed funded in-kind and is not charged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FREQUENCY_MONTHS = {"monthly": 1, "quarterly": 3, "annual": 12, "annually": 12, "yearly": 12}


def frequency_to_months(frequency: str) -> int:
    """Map a calendar frequency name to an interval in months."""
    try:
        return _FREQUENCY_MONTHS[frequency.lower()]
    except KeyError as exc:  # pragma: no cover - guarded by config
        raise ValueError(
            f"Unknown rebalance frequency {frequency!r}; expected one of {sorted(_FREQUENCY_MONTHS)}"
        ) from exc


def is_calendar_rebalance(periods_since_first: int, frequency: str) -> bool:
    """True on the first investable period and every ``interval`` months after."""
    interval = frequency_to_months(frequency)
    return periods_since_first % interval == 0


def exceeds_band(current: pd.Series, target: pd.Series, band: float) -> bool:
    """True if any asset's drifted weight is more than ``band`` from target."""
    return bool((current.subtract(target, fill_value=0.0).abs() > band).any())


def turnover_and_cost(current: pd.Series, target: pd.Series, cost_bps: float) -> tuple[float, float]:
    """Return ``(one_way_turnover, cost_fraction)`` for a trade ``current → target``.

    See the module docstring for the convention. ``cost_fraction`` is the
    fraction of portfolio value lost to transaction costs at the rebalance.
    """
    delta = float(current.subtract(target, fill_value=0.0).abs().sum())
    one_way_turnover = 0.5 * delta
    cost_fraction = (cost_bps / 10_000.0) * delta
    return one_way_turnover, cost_fraction


def drift_weights(weights: pd.Series, asset_returns: pd.Series) -> tuple[pd.Series, float]:
    """Evolve weights over one period given per-asset returns.

    Returns ``(new_weights, portfolio_return)`` where the portfolio return is
    ``Σ_i w_i r_i`` and the new weights are the value-weighted (drifted)
    proportions after the period: ``w_i (1+r_i) / (1 + Σ w r)``.
    """
    port_ret = float((weights * asset_returns).sum())
    grown = weights * (1.0 + asset_returns)
    denom = 1.0 + port_ret
    if denom <= 0:  # pragma: no cover - total wipeout guard
        new_weights = grown  # leave un-normalised; downstream handles it
    else:
        new_weights = grown / denom
    return new_weights, port_ret
