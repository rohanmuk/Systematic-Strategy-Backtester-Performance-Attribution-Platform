"""Assemble the headline analytics for a backtest result.

This is the glue between the raw engine output and the reporting layer: it
calls into :mod:`backtester.metrics` and :mod:`backtester.attribution` to build
a single, ordered summary (absolute + benchmark-relative + cost) plus the
rolling series used by the tearsheet.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import pandas as pd

from . import metrics
from .attribution import compute_attribution, turnover_summary
from .engine import BacktestResult


def summarize(
    result: BacktestResult,
    benchmark: pd.Series,
    rf: pd.Series,
) -> "OrderedDict[str, float]":
    """Return an ordered dict of headline metrics for one strategy.

    ``benchmark`` and ``rf`` are monthly return series; they are aligned to the
    result's window inside each metric.
    """
    r = result.returns
    rf = rf.reindex(r.index).fillna(0.0)
    bench = benchmark.reindex(r.index).dropna()

    turn = turnover_summary(result.turnover, result.costs)

    out: "OrderedDict[str, float]" = OrderedDict()
    # Absolute
    out["CAGR"] = metrics.cagr(r)
    out["Ann. Return (arith)"] = metrics.annualized_return_arithmetic(r)
    out["Ann. Volatility"] = metrics.annualized_volatility(r)
    out["Sharpe"] = metrics.sharpe_ratio(r, rf)
    out["Sortino"] = metrics.sortino_ratio(r, rf)
    out["Max Drawdown"] = metrics.max_drawdown(r)
    out["Calmar"] = metrics.calmar_ratio(r)
    # Benchmark-relative
    out["Active Return"] = metrics.active_return(r, bench)
    out["Tracking Error"] = metrics.tracking_error(r, bench)
    out["Information Ratio"] = metrics.information_ratio(r, bench)
    out["Beta"] = metrics.beta(r, bench)
    out["Alpha"] = metrics.alpha(r, bench, rf)
    out["Up Capture"] = metrics.up_capture(r, bench)
    out["Down Capture"] = metrics.down_capture(r, bench)
    # Cost / trading
    out["Annual Turnover"] = turn["annual_turnover"]
    out["Annual Cost Drag"] = turn["annual_cost_drag"]
    return out


def rolling_frame(
    result: BacktestResult, rf: pd.Series, windows=(12, 36)
) -> Dict[str, pd.Series]:
    """Rolling return / vol / Sharpe series keyed like ``'return_12m'``."""
    r = result.returns
    rf = rf.reindex(r.index).fillna(0.0)
    series: Dict[str, pd.Series] = {}
    for w in windows:
        series[f"return_{w}m"] = metrics.rolling_return(r, w)
        series[f"vol_{w}m"] = metrics.rolling_volatility(r, w)
        series[f"sharpe_{w}m"] = metrics.rolling_sharpe(r, w, rf)
    return series


def attribution_tables(result: BacktestResult, asset_returns: pd.DataFrame):
    """Convenience wrapper returning (contribution_to_return, contribution_to_risk)."""
    attr = compute_attribution(result.weights, asset_returns)
    return attr.to_return, attr.to_risk


def comparison_table(
    results: List[BacktestResult],
    benchmark: pd.Series,
    rf: pd.Series,
    include_benchmark_row: bool = True,
    benchmark_name: str = "Benchmark",
) -> pd.DataFrame:
    """Strategy-by-metric comparison table (strategies as rows).

    All strategies are aligned to their common overlapping window before
    metrics are computed, so the comparison is apples-to-apples.
    """
    common = None
    for res in results:
        common = res.returns.index if common is None else common.intersection(res.returns.index)
    common = common.intersection(benchmark.index)

    rows: "OrderedDict[str, OrderedDict[str, float]]" = OrderedDict()
    bench_common = benchmark.reindex(common)
    for res in results:
        clipped = _clip_result(res, common)
        rows[res.name] = summarize(clipped, bench_common, rf)

    if include_benchmark_row:
        rows[benchmark_name] = _benchmark_summary(bench_common, rf)

    return pd.DataFrame(rows).T


def _clip_result(result: BacktestResult, index: pd.Index) -> BacktestResult:
    return BacktestResult(
        name=result.name,
        returns=result.returns.reindex(index),
        gross_returns=result.gross_returns.reindex(index),
        weights=result.weights.reindex(index),
        turnover=result.turnover.reindex(index).fillna(0.0),
        costs=result.costs.reindex(index).fillna(0.0),
        config=result.config,
    )


def _benchmark_summary(bench: pd.Series, rf: pd.Series) -> "OrderedDict[str, float]":
    """Summary row for the benchmark itself (self-relative fields are trivial)."""
    rf = rf.reindex(bench.index).fillna(0.0)
    out: "OrderedDict[str, float]" = OrderedDict()
    out["CAGR"] = metrics.cagr(bench)
    out["Ann. Return (arith)"] = metrics.annualized_return_arithmetic(bench)
    out["Ann. Volatility"] = metrics.annualized_volatility(bench)
    out["Sharpe"] = metrics.sharpe_ratio(bench, rf)
    out["Sortino"] = metrics.sortino_ratio(bench, rf)
    out["Max Drawdown"] = metrics.max_drawdown(bench)
    out["Calmar"] = metrics.calmar_ratio(bench)
    out["Active Return"] = 0.0
    out["Tracking Error"] = 0.0
    out["Information Ratio"] = float("nan")
    out["Beta"] = 1.0
    out["Alpha"] = 0.0
    out["Up Capture"] = 1.0
    out["Down Capture"] = 1.0
    out["Annual Turnover"] = float("nan")
    out["Annual Cost Drag"] = 0.0
    return out
