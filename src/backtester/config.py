"""Typed configuration model and YAML loader.

A single YAML file fully specifies a backtest: the asset universe, the data
window, the allocation strategy and its parameters, the rebalancing policy,
transaction costs, and the benchmark. Anything omitted falls back to a
sensible, explicitly-documented default defined here (not hidden in the YAML),
so configs stay short and the defaults are reviewable in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (single source of truth). Referenced by the loader below.
# ---------------------------------------------------------------------------

#: Diversified ETF universe. Adjusted-close (total-return) proxies for the
#: major liquid asset classes a multi-asset analyst allocates across.
DEFAULT_UNIVERSE: List[str] = [
    "VTI",  # US total equity market
    "VEA",  # Developed ex-US equity
    "VWO",  # Emerging-market equity
    "AGG",  # US aggregate bonds
    "TIP",  # US TIPS (inflation-linked)
    "GLD",  # Gold
    "VNQ",  # US REITs
    "BIL",  # 1-3 month T-bills (cash / risk-free proxy)
]

#: Cash / risk-free proxy ticker. Its monthly return is the risk-free rate
#: used in Sharpe/Sortino/alpha, and target-vol parks residual weight here.
DEFAULT_RISK_FREE = "BIL"

#: Default benchmark: the canonical 60% US equity / 40% US bonds portfolio,
#: rebalanced monthly. Weights must sum to 1.
DEFAULT_BENCHMARK_WEIGHTS: Dict[str, float] = {"VTI": 0.60, "AGG": 0.40}


@dataclass
class DataConfig:
    """Data window and frequency.

    ``start=None`` means "use the longest common history across the universe"
    (the first month in which *every* ticker has data). ``end=None`` means
    "up to today". Frequency is fixed to monthly for this engine.
    """

    start: Optional[str] = None
    end: Optional[str] = None
    frequency: str = "monthly"
    cache: bool = True


@dataclass
class StrategyConfig:
    """Which weighting rule to run and its parameters.

    ``type`` keys into ``backtester.strategies.STRATEGY_REGISTRY``. ``params``
    is passed through to that strategy function (e.g. lookback windows, target
    vol, static weights, glide-path knots).
    """

    type: str = "equal_weight"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RebalanceConfig:
    """Rebalancing policy and trading frictions.

    ``method`` is ``"calendar"`` (rebalance every N months per ``frequency``)
    or ``"threshold"`` (rebalance only when any asset's actual weight drifts
    more than ``band`` from its target). ``"buy_and_hold"`` never rebalances
    after the initial allocation. ``cost_bps`` is charged per unit of one-way
    turnover.
    """

    method: str = "calendar"
    frequency: str = "quarterly"  # monthly | quarterly | annual
    band: float = 0.05
    cost_bps: float = 10.0


@dataclass
class BenchmarkConfig:
    """Benchmark portfolio for relative analytics.

    Defaults to a static 60/40 (VTI/AGG) rebalanced monthly. Any static weight
    set can be supplied; weights are renormalised to sum to 1 if needed.
    """

    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BENCHMARK_WEIGHTS))
    rebalance_frequency: str = "monthly"
    name: str = "60/40"


@dataclass
class BacktestConfig:
    """Top-level, fully-resolved configuration for one backtest run."""

    name: str
    description: str = ""
    tickers: List[str] = field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    risk_free: str = DEFAULT_RISK_FREE
    initial_capital: float = 10_000.0
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def __post_init__(self) -> None:
        if self.risk_free not in self.tickers:
            raise ValueError(
                f"risk_free proxy {self.risk_free!r} must be part of the universe {self.tickers}"
            )


def _build_dataclass(cls, data: Optional[Dict[str, Any]]):
    """Instantiate a dataclass from a dict, ignoring unknown keys with a warning."""
    data = data or {}
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(data) - field_names
    if unknown:
        logger.warning("Ignoring unknown %s keys: %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in data.items() if k in field_names})


def load_config(path: str | Path) -> BacktestConfig:
    """Parse a YAML file into a validated :class:`BacktestConfig`.

    Missing sections fall back to the module-level defaults. ``name`` defaults
    to the file stem if not given in the YAML.
    """
    path = Path(path)
    with path.open("r") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    universe = raw.get("universe", {}) or {}
    tickers = universe.get("tickers", list(DEFAULT_UNIVERSE))
    risk_free = universe.get("risk_free", DEFAULT_RISK_FREE)

    cfg = BacktestConfig(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        tickers=list(tickers),
        risk_free=risk_free,
        initial_capital=float(raw.get("initial_capital", 10_000.0)),
        data=_build_dataclass(DataConfig, raw.get("data")),
        strategy=_build_dataclass(StrategyConfig, raw.get("strategy")),
        rebalance=_build_dataclass(RebalanceConfig, raw.get("rebalance")),
        benchmark=_build_dataclass(BenchmarkConfig, raw.get("benchmark")),
    )
    logger.info("Loaded config %r (strategy=%s) from %s", cfg.name, cfg.strategy.type, path)
    return cfg
