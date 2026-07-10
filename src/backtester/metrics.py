"""Performance & risk metrics, implemented from first principles.

Every formula here is written out explicitly (no empyrical/quantstats at
runtime) so each number in a tearsheet can be defended in an interview. The
test-suite cross-checks a subset against those libraries, but they are never a
runtime dependency.

Conventions
-----------
* Inputs are **simple** (arithmetic) periodic returns, monthly by default, as a
  pandas ``Series`` indexed by date.
* ``periods_per_year`` is the annualisation factor (12 for monthly).
* ``rf`` (risk-free) may be a scalar *per-period* rate or a ``Series`` aligned
  to ``returns``; it is broadcast to per-period excess returns before use.
* Volatility uses the **sample** standard deviation (``ddof=1``).
* CAGR / drawdown are **geometric** (compounded); Sharpe/Sortino/IR use the
  arithmetic mean of periodic (excess/active) returns annualised by the usual
  ``mean * P`` / ``std * sqrt(P)`` scaling. This split is the industry
  convention and is stated wherever it matters.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12  # monthly engine

RfLike = Union[float, pd.Series]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _excess_returns(returns: pd.Series, rf: RfLike) -> pd.Series:
    """Per-period excess return over the risk-free rate.

    ``rf`` as a Series is aligned on the index; as a scalar it is the
    per-period rate subtracted from every observation.
    """
    if isinstance(rf, pd.Series):
        rf = rf.reindex(returns.index).fillna(0.0)
    return returns - rf


def wealth_index(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compounded growth of `initial` invested at the start: ``∏(1+r_t)``."""
    return initial * (1.0 + returns).cumprod()


# ---------------------------------------------------------------------------
# Absolute performance
# ---------------------------------------------------------------------------

def cagr(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Compound annual growth rate.

    ``CAGR = (∏(1+r_t))^(P / n) − 1`` where ``n`` is the number of periods and
    ``P`` the periods-per-year. This is the geometric annualised return.
    """
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        return float("nan")
    total_growth = float((1.0 + returns).prod())
    if total_growth <= 0:  # total wipe-out; CAGR undefined as a real number
        return -1.0
    return total_growth ** (periods_per_year / n) - 1.0


def annualized_return_arithmetic(
    returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Arithmetic annualised return: ``mean(r) * P`` (used inside Sharpe/alpha)."""
    return float(returns.dropna().mean() * periods_per_year)


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualised standard deviation: ``std(r, ddof=1) * sqrt(P)``."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, rf: RfLike = 0.0, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualised Sharpe ratio.

    ``Sharpe = mean(r − rf) / std(r − rf) * sqrt(P)`` — mean excess return per
    unit of total volatility, annualised. Uses sample std (``ddof=1``).
    """
    excess = _excess_returns(returns, rf).dropna()
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-13:  # degenerate zero-vol series
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series, mar: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualised downside deviation below a minimum acceptable return (MAR).

    ``DD = sqrt( mean( min(r − MAR, 0)^2 ) ) * sqrt(P)``. Note the mean is over
    **all** periods (not just the down periods) — the standard Sortino
    definition — so DD penalises the frequency of shortfalls, not only their
    depth.
    """
    returns = returns.dropna()
    downside = np.minimum(returns - mar, 0.0)
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    rf: RfLike = 0.0,
    mar: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio: annualised excess return / downside deviation."""
    excess = _excess_returns(returns, rf).dropna()
    ann_excess = excess.mean() * periods_per_year
    dd = downside_deviation(returns, mar=mar, periods_per_year=periods_per_year)
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(ann_excess / dd)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Path of drawdowns: ``wealth / running_peak − 1`` (≤ 0)."""
    wealth = wealth_index(returns)
    running_peak = wealth.cummax()
    return wealth / running_peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline, as a **positive** fraction (0.34 = −34%)."""
    dd = drawdown_series(returns)
    if dd.empty:
        return float("nan")
    return float(-dd.min())


def max_drawdown_dates(returns: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(peak_date, trough_date) of the maximum drawdown."""
    wealth = wealth_index(returns)
    dd = wealth / wealth.cummax() - 1.0
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    return peak, trough


def calmar_ratio(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Calmar ratio: ``CAGR / max_drawdown`` (return per unit of worst loss)."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr(returns, periods_per_year) / mdd


# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------

def rolling_return(returns: pd.Series, window: int) -> pd.Series:
    """Trailing compounded return over a rolling window of ``window`` periods."""
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def rolling_volatility(
    returns: pd.Series, window: int, periods_per_year: int = PERIODS_PER_YEAR
) -> pd.Series:
    """Trailing annualised volatility over a rolling window."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    rf: RfLike = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Trailing annualised Sharpe ratio over a rolling window."""
    excess = _excess_returns(returns, rf)
    mean = excess.rolling(window).mean()
    sd = excess.rolling(window).std(ddof=1)
    return mean / sd * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------

def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    return joined.iloc[:, 0], joined.iloc[:, 1]


def active_return(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualised active return: ``mean(r_p − r_b) * P`` (arithmetic)."""
    rp, rb = _align(returns, benchmark)
    return float((rp - rb).mean() * periods_per_year)


def tracking_error(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Annualised tracking error: ``std(r_p − r_b, ddof=1) * sqrt(P)``."""
    rp, rb = _align(returns, benchmark)
    return float((rp - rb).std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series, benchmark: pd.Series, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    """Information ratio: annualised active return / tracking error."""
    te = tracking_error(returns, benchmark, periods_per_year)
    if te == 0 or np.isnan(te):
        return float("nan")
    return active_return(returns, benchmark, periods_per_year) / te


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """CAPM beta: ``Cov(r_p, r_b) / Var(r_b)`` (sample moments)."""
    rp, rb = _align(returns, benchmark)
    if len(rp) < 2:
        return float("nan")
    var_b = np.var(rb, ddof=1)
    if var_b == 0:
        return float("nan")
    return float(np.cov(rp, rb, ddof=1)[0, 1] / var_b)


def alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    rf: RfLike = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> float:
    """Annualised Jensen's alpha from the CAPM regression.

    ``alpha = mean(r_p − rf) − beta * mean(r_b − rf)``, then annualised by ``* P``.
    It is the average excess return not explained by benchmark exposure.
    """
    rp, rb = _align(returns, benchmark)
    b = beta(rp, rb)
    ex_p = _excess_returns(rp, rf)
    ex_b = _excess_returns(rb, rf)
    per_period_alpha = ex_p.mean() - b * ex_b.mean()
    return float(per_period_alpha * periods_per_year)


def _geometric_mean(returns: pd.Series) -> float:
    """Geometric mean per-period return: ``(∏(1+r))^(1/n) − 1``."""
    n = len(returns)
    if n == 0:
        return float("nan")
    return float((1.0 + returns).prod() ** (1.0 / n) - 1.0)


def up_capture(returns: pd.Series, benchmark: pd.Series) -> float:
    """Up-market capture ratio (Morningstar-style, geometric).

    Over the months where the **benchmark** was up, the ratio of the strategy's
    geometric-mean return to the benchmark's. >1 means the strategy outperforms
    in up markets.
    """
    rp, rb = _align(returns, benchmark)
    mask = rb > 0
    if mask.sum() == 0:
        return float("nan")
    g_b = _geometric_mean(rb[mask])
    if g_b == 0:
        return float("nan")
    return _geometric_mean(rp[mask]) / g_b


def down_capture(returns: pd.Series, benchmark: pd.Series) -> float:
    """Down-market capture ratio (geometric).

    Over the months where the benchmark was down, the ratio of the strategy's
    geometric-mean return to the benchmark's. <1 (and positive) means the
    strategy loses less than the benchmark in down markets.
    """
    rp, rb = _align(returns, benchmark)
    mask = rb < 0
    if mask.sum() == 0:
        return float("nan")
    g_b = _geometric_mean(rb[mask])
    if g_b == 0:
        return float("nan")
    return _geometric_mean(rp[mask]) / g_b
