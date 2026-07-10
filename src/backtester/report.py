"""End-to-end run orchestration and report rendering.

``run_config`` loads data, runs the strategy and its benchmark, and returns a
tidy bundle. ``build_report`` / ``build_comparison_report`` turn those bundles
into a CSV summary, a tearsheet PNG, and a self-contained HTML report with the
charts embedded as base64 (so a single .html file is fully portable).
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from .analysis import attribution_tables, comparison_table, summarize
from .config import BacktestConfig
from .data import PriceData, load_price_data
from .engine import BacktestResult, run_backtest, run_benchmark_returns
from .plotting import comparison_equity_chart, save_figure, tearsheet

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

# Metrics reported as percentages (everything else is a unitless ratio).
_PERCENT_METRICS = {
    "CAGR",
    "Ann. Return (arith)",
    "Ann. Volatility",
    "Max Drawdown",
    "Active Return",
    "Tracking Error",
    "Alpha",
    "Annual Turnover",
    "Annual Cost Drag",
}


@dataclass
class RunBundle:
    """Everything produced by running one config against loaded data."""

    result: BacktestResult
    benchmark: pd.Series
    rf: pd.Series
    price_data: PriceData
    summary: "dict[str, float]"


def run_config(
    cfg: BacktestConfig,
    price_data: Optional[PriceData] = None,
    warmup: int = 12,
) -> RunBundle:
    """Load data (if not supplied), run the strategy + benchmark, summarise."""
    if price_data is None:
        price_data = load_price_data(
            cfg.tickers, start=cfg.data.start, end=cfg.data.end, cache=cfg.data.cache
        )
    returns = price_data.monthly_returns

    result = run_backtest(returns, cfg, warmup=warmup)
    benchmark = run_benchmark_returns(returns, cfg.benchmark, cfg.risk_free, warmup=warmup)
    rf = returns[cfg.risk_free]
    summary = summarize(result, benchmark, rf)
    return RunBundle(result=result, benchmark=benchmark, rf=rf, price_data=price_data, summary=dict(summary))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "—"
    if metric in _PERCENT_METRICS:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}"


def format_summary_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Format a raw (numeric) summary table into display strings."""
    out = df.copy()
    for metric in out.columns:
        out[metric] = [format_value(metric, v) for v in out[metric]]
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto; max-width: 1100px; color: #1a1a1a; }}
 h1 {{ font-size: 1.6rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
 table {{ border-collapse: collapse; font-size: .85rem; margin: 1rem 0; }}
 th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
 th {{ background: #f4f4f4; }} td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; height: auto; }} .meta {{ color: #666; font-size: .85rem; }}
</style></head><body>
<h1>{title}</h1>
<p class="meta">{subtitle}</p>
{body}
</body></html>"""


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(border=0, escape=False)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def build_report(bundle: RunBundle, out_dir: Path = REPORTS_DIR) -> Dict[str, Path]:
    """Write CSV summary, tearsheet PNG, and an HTML report for one strategy."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(bundle.result.name)

    # Summary CSV (raw numbers).
    summary_df = pd.DataFrame({bundle.result.name: bundle.summary})
    csv_path = out_dir / f"{slug}_summary.csv"
    summary_df.to_csv(csv_path)

    # Attribution tables.
    ctr, ctrisk = attribution_tables(bundle.result, bundle.price_data.monthly_returns)

    # Tearsheet.
    fig = tearsheet(bundle.result, bundle.benchmark, bundle.rf, bundle.price_data.monthly_returns)
    png_path = save_figure(fig, fig_dir / f"{slug}_tearsheet.png")

    # HTML (re-render tearsheet to embed inline so the file is portable).
    fig2 = tearsheet(bundle.result, bundle.benchmark, bundle.rf, bundle.price_data.monthly_returns)
    img_b64 = _fig_to_base64(fig2)

    fmt_summary = format_summary_frame(summary_df.T).T
    body = (
        f"<h2>Headline metrics vs {bundle.result.config.benchmark.name}</h2>"
        f"{_table_html(fmt_summary)}"
        f"<h2>Tearsheet</h2><img src='data:image/png;base64,{img_b64}'/>"
        f"<h2>Contribution to return</h2>{_table_html(ctr.round(4))}"
        f"<h2>Contribution to risk</h2>{_table_html(ctrisk.round(4))}"
    )
    html = _HTML_TEMPLATE.format(
        title=f"Backtest report — {bundle.result.name}",
        subtitle=(
            f"{bundle.result.returns.index.min().date()} → {bundle.result.returns.index.max().date()} · "
            f"strategy={bundle.result.config.strategy.type} · "
            f"rebalance={bundle.result.config.rebalance.method}/{bundle.result.config.rebalance.frequency} · "
            f"cost={bundle.result.config.rebalance.cost_bps:.0f}bps"
        ),
        body=body,
    )
    html_path = out_dir / f"{slug}_report.html"
    html_path.write_text(html)

    logger.info("Wrote report for %r -> %s", bundle.result.name, html_path)
    return {"csv": csv_path, "png": png_path, "html": html_path}


def build_comparison_report(
    bundles: List[RunBundle], out_dir: Path = REPORTS_DIR, name: str = "comparison"
) -> Dict[str, Path]:
    """Write a multi-strategy comparison: CSV table, equity chart, HTML."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [b.result for b in bundles]
    benchmark = bundles[0].benchmark
    rf = bundles[0].rf

    table = comparison_table(results, benchmark, rf, benchmark_name=bundles[0].result.config.benchmark.name)
    csv_path = out_dir / f"{name}_summary.csv"
    table.to_csv(csv_path)

    fig = comparison_equity_chart(results, benchmark)
    png_path = save_figure(fig, fig_dir / f"{name}_equity.png")

    fig2 = comparison_equity_chart(results, benchmark)
    img_b64 = _fig_to_base64(fig2)

    body = (
        f"<h2>Growth of $1</h2><img src='data:image/png;base64,{img_b64}'/>"
        f"<h2>Metric comparison</h2>{_table_html(format_summary_frame(table))}"
    )
    html = _HTML_TEMPLATE.format(
        title="Strategy comparison",
        subtitle=f"{len(results)} strategies · benchmark={bundles[0].result.config.benchmark.name}",
        body=body,
    )
    html_path = out_dir / f"{name}_report.html"
    html_path.write_text(html)

    logger.info("Wrote comparison report -> %s", html_path)
    return {"csv": csv_path, "png": png_path, "html": html_path}


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
