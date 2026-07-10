"""Command-line interface.

    python -m backtester run     --config configs/60_40.yaml
    python -m backtester compare --configs configs/*.yaml
    python -m backtester fetch   --config configs/60_40.yaml   # prime the data cache

The console-script entry point ``backtester`` (see pyproject) maps to
:func:`main`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

from .config import load_config
from .data import load_price_data
from .report import build_comparison_report, build_report, format_summary_frame, run_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # yfinance is chatty; keep it quiet unless -v.
    if not verbose:
        logging.getLogger("yfinance").setLevel(logging.WARNING)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    bundle = run_config(cfg, warmup=args.warmup)
    summary_df = pd.DataFrame({cfg.name: bundle.summary})
    print(f"\n=== {cfg.name} ===")
    print(format_summary_frame(summary_df.T).T.to_string())

    if not args.no_report:
        paths = build_report(bundle, out_dir=Path(args.out))
        print("\nWrote:")
        for kind, path in paths.items():
            print(f"  {kind:4s} -> {path}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    config_paths = _expand_configs(args.configs)
    if not config_paths:
        print("No config files matched.", file=sys.stderr)
        return 2
    print(f"Comparing {len(config_paths)} strategies:")
    for p in config_paths:
        print(f"  - {p}")

    bundles = []
    shared_data = None
    base_tickers = None
    for path in config_paths:
        cfg = load_config(path)
        # Reuse the loaded panel across configs that share a universe.
        if base_tickers is None:
            shared_data = load_price_data(cfg.tickers, start=cfg.data.start, end=cfg.data.end)
            base_tickers = sorted(cfg.tickers)
        price_data = shared_data if sorted(cfg.tickers) == base_tickers else None
        bundles.append(run_config(cfg, price_data=price_data, warmup=args.warmup))

    table = build_comparison_report(bundles, out_dir=Path(args.out))
    # Rebuild the display table for stdout.
    from .analysis import comparison_table

    disp = comparison_table(
        [b.result for b in bundles],
        bundles[0].benchmark,
        bundles[0].rf,
        benchmark_name=bundles[0].result.config.benchmark.name,
    )
    print("\n=== Comparison ===")
    print(format_summary_frame(disp).to_string())
    print("\nWrote:")
    for kind, path in table.items():
        print(f"  {kind:4s} -> {path}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    data = load_price_data(
        cfg.tickers, start=cfg.data.start, end=cfg.data.end, force_refresh=args.refresh
    )
    print(
        f"Cached {data.monthly_returns.shape[1]} assets, "
        f"{data.start.date()} -> {data.end.date()} ({len(data.monthly_returns)} months)."
    )
    if data.dropped_tickers:
        print(f"Dropped (insufficient history): {data.dropped_tickers}")
    return 0


def _expand_configs(patterns: List[str]) -> List[Path]:
    """Accept explicit files, globs (pre- or post-shell-expansion), or dirs."""
    paths: List[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.yaml")))
        elif any(ch in pat for ch in "*?["):
            paths.extend(sorted(Path().glob(pat)))
        elif p.exists():
            paths.append(p)
    # De-dup, preserve order.
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backtester", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--warmup", type=int, default=12, help="months reserved to form the first estimate")
    parser.add_argument("--out", default="reports", help="output directory for reports")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run and report a single strategy config")
    p_run.add_argument("--config", required=True, help="path to a strategy YAML")
    p_run.add_argument("--no-report", action="store_true", help="print metrics only; skip files")
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="run and compare several configs")
    p_cmp.add_argument("--configs", required=True, nargs="+", help="YAML paths, globs, or a directory")
    p_cmp.set_defaults(func=_cmd_compare)

    p_fetch = sub.add_parser("fetch", help="download and cache the data for a config")
    p_fetch.add_argument("--config", required=True)
    p_fetch.add_argument("--refresh", action="store_true", help="force re-download")
    p_fetch.set_defaults(func=_cmd_fetch)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
