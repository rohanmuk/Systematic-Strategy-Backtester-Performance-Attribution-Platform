"""Tearsheet and comparison charts (matplotlib, headless ``Agg`` backend).

Every function draws onto a provided ``Axes`` so panels compose into a single
tearsheet figure; convenience wrappers save standalone PNGs for embedding in
the README / HTML report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # no display; render straight to file

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics  # noqa: E402
from .analysis import rolling_frame  # noqa: E402
from .engine import BacktestResult  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)


def plot_equity_curve(returns: pd.Series, benchmark: Optional[pd.Series], ax: plt.Axes) -> None:
    wealth = (1.0 + returns).cumprod()
    ax.plot(wealth.index, wealth.values, label=returns.name or "Strategy", lw=1.6)
    if benchmark is not None:
        bw = (1.0 + benchmark.reindex(returns.index).fillna(0.0)).cumprod()
        ax.plot(bw.index, bw.values, label=benchmark.name or "Benchmark", lw=1.2, ls="--", color="grey")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (log scale)")
    ax.legend(fontsize=8)


def plot_drawdown(returns: pd.Series, ax: plt.Axes) -> None:
    dd = metrics.drawdown_series(returns)
    ax.fill_between(dd.index, dd.values * 100.0, 0.0, color="crimson", alpha=0.4)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("%")


def plot_rolling_sharpe(returns: pd.Series, rf: pd.Series, ax: plt.Axes, window: int = 36) -> None:
    rs = metrics.rolling_sharpe(returns, window, rf)
    ax.plot(rs.index, rs.values, lw=1.3, color="teal")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(f"Rolling {window}-month Sharpe")


def plot_weights_area(weights: pd.DataFrame, ax: plt.Axes) -> None:
    w = weights.clip(lower=0.0)
    ax.stackplot(w.index, *[w[c].values for c in w.columns], labels=list(w.columns))
    ax.set_ylim(0, 1)
    ax.set_title("Weights over time")
    ax.legend(fontsize=7, ncol=2, loc="upper left")


def plot_return_distribution(returns: pd.Series, ax: plt.Axes) -> None:
    ax.hist(returns.values * 100.0, bins=30, color="steelblue", alpha=0.8)
    ax.axvline(float(returns.mean() * 100.0), color="black", ls="--", lw=1, label="mean")
    ax.set_title("Monthly return distribution (%)")
    ax.legend(fontsize=8)


def plot_correlation_heatmap(asset_returns: pd.DataFrame, ax: plt.Axes) -> None:
    corr = asset_returns.corr()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    values = corr.values
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(values[i, j]) > 0.6 else "black")
    ax.grid(False)
    ax.set_title("Asset return correlations")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def tearsheet(
    result: BacktestResult,
    benchmark: pd.Series,
    rf: pd.Series,
    asset_returns: pd.DataFrame,
) -> plt.Figure:
    """Compose the six-panel tearsheet for a single strategy."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(f"Tearsheet — {result.name}", fontsize=13, y=0.995)

    plot_equity_curve(result.returns, benchmark, axes[0, 0])
    plot_drawdown(result.returns, axes[0, 1])
    plot_rolling_sharpe(result.returns, rf, axes[1, 0])
    plot_weights_area(result.weights, axes[1, 1])
    plot_return_distribution(result.returns, axes[2, 0])
    held_assets = [c for c in asset_returns.columns if c in result.weights.columns]
    plot_correlation_heatmap(asset_returns[held_assets].reindex(result.returns.index), axes[2, 1])

    fig.tight_layout()
    return fig


def comparison_equity_chart(
    results: List[BacktestResult], benchmark: pd.Series
) -> plt.Figure:
    """Overlay net equity curves for several strategies plus the benchmark."""
    fig, ax = plt.subplots(figsize=(11, 6))
    common = None
    for res in results:
        common = res.returns.index if common is None else common.intersection(res.returns.index)
    for res in results:
        w = (1.0 + res.returns.reindex(common).fillna(0.0)).cumprod()
        ax.plot(w.index, w.values, lw=1.4, label=res.name)
    bw = (1.0 + benchmark.reindex(common).fillna(0.0)).cumprod()
    ax.plot(bw.index, bw.values, lw=1.6, ls="--", color="black", label=benchmark.name or "Benchmark")
    ax.set_yscale("log")
    ax.set_title("Strategy comparison — growth of $1 (log scale)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
