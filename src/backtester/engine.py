"""The backtest engine: turn a config + monthly returns into a track record.

Timing (the no-lookahead contract)
----------------------------------
The engine works on month-end return observations. A decision made at the end
of month ``t`` may use return data **only through month t**, and the resulting
weights are held over month ``t+1``. Concretely, target weights at index ``t``
are computed from ``returns.iloc[:t+1]`` and earn ``returns.iloc[t+1]``. There
is no way for information from ``t+1`` to influence the weights that earn its
return — this is asserted in the tests.

Between rebalances, weights **drift** with realised asset returns. On a
rebalance the portfolio is traded back to the strategy's current target and a
transaction cost (see :mod:`backtester.rebalance`) is charged to the following
period. The very first allocation is assumed funded in-kind (no cost).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig, BenchmarkConfig
from .rebalance import (
    drift_weights,
    exceeds_band,
    is_calendar_rebalance,
    turnover_and_cost,
)
from .strategies import StrategyContext, get_strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Full output of a single backtest, aligned on the investable months."""

    name: str
    returns: pd.Series  # net-of-cost monthly portfolio returns
    gross_returns: pd.Series  # before transaction costs
    weights: pd.DataFrame  # weights held during each period (rows=month, cols=asset)
    turnover: pd.Series  # one-way turnover attributed to each period
    costs: pd.Series  # cost drag (fraction) attributed to each period
    config: BacktestConfig

    @property
    def wealth(self) -> pd.Series:
        """Growth of 1 unit (net of costs)."""
        return (1.0 + self.returns).cumprod()

    @property
    def initial_capital(self) -> float:
        return self.config.initial_capital


def run_backtest(
    returns: pd.DataFrame,
    cfg: BacktestConfig,
    warmup: int = 12,
) -> BacktestResult:
    """Simulate ``cfg``'s strategy over ``returns`` (monthly, columns=assets).

    ``warmup`` months of history are reserved to form the first estimate; the
    investable track record starts at month ``warmup``. Covariance/vol-based
    strategies use whatever history exists if it is shorter than their lookback.
    """
    dates = returns.index
    n = len(dates)
    if n <= warmup + 1:
        raise ValueError(
            f"Not enough history ({n} months) for warmup={warmup}; need at least {warmup + 2}."
        )

    strategy = get_strategy(cfg.strategy.type)
    params = dict(cfg.strategy.params)
    method = cfg.rebalance.method
    start_date = dates[warmup - 1]  # first decision date (used by the glide path)

    def make_ctx(decision_date: pd.Timestamp, prior: Optional[pd.Series]) -> StrategyContext:
        return StrategyContext(
            columns=list(returns.columns),
            risk_free=cfg.risk_free,
            date=decision_date,
            start_date=start_date,
            prior_weights=prior,
        )

    # --- initial allocation (funded in-kind, no cost) -----------------------
    ctx0 = make_ctx(start_date, None)
    current = strategy(returns.iloc[:warmup], ctx0, **params)
    _validate_weights(current, cfg.name, start_date)

    idx: list[pd.Timestamp] = []
    net_returns: list[float] = []
    gross_returns: list[float] = []
    held: list[pd.Series] = []
    turnovers: list[float] = []
    costs: list[float] = []

    pending_turnover = 0.0
    pending_cost = 0.0
    periods_since_first_decision = 0

    for t in range(warmup, n):
        r_t = returns.iloc[t]

        # Period t: hold `current`, drift it, earn its return (net of any cost
        # from a trade executed at this period's start).
        drifted, gross_ret = drift_weights(current, r_t)
        net_ret = gross_ret - pending_cost

        idx.append(dates[t])
        gross_returns.append(gross_ret)
        net_returns.append(net_ret)
        held.append(current.copy())
        turnovers.append(pending_turnover)
        costs.append(pending_cost)

        pending_turnover, pending_cost = 0.0, 0.0

        # Decision at end of month t (data through t only) for month t+1.
        if t < n - 1:
            periods_since_first_decision += 1
            decision_date = dates[t]
            target = strategy(returns.iloc[: t + 1], make_ctx(decision_date, drifted), **params)
            _validate_weights(target, cfg.name, decision_date)

            if _should_rebalance(method, cfg, periods_since_first_decision, drifted, target):
                pending_turnover, pending_cost = turnover_and_cost(
                    drifted, target, cfg.rebalance.cost_bps
                )
                current = target
            else:
                current = drifted
        else:
            current = drifted

    result = BacktestResult(
        name=cfg.name,
        returns=pd.Series(net_returns, index=idx, name=cfg.name),
        gross_returns=pd.Series(gross_returns, index=idx, name=cfg.name),
        weights=pd.DataFrame(held, index=idx),
        turnover=pd.Series(turnovers, index=idx, name="turnover"),
        costs=pd.Series(costs, index=idx, name="cost"),
        config=cfg,
    )
    logger.info(
        "Backtest %r: %d months, CAGR-ready. Total one-way turnover=%.2f, cost drag=%.4f.",
        cfg.name,
        len(idx),
        float(result.turnover.sum()),
        float(result.costs.sum()),
    )
    return result


def _should_rebalance(
    method: str,
    cfg: BacktestConfig,
    periods_since_first_decision: int,
    current: pd.Series,
    target: pd.Series,
) -> bool:
    if method == "buy_and_hold":
        return False
    if method == "calendar":
        return is_calendar_rebalance(periods_since_first_decision, cfg.rebalance.frequency)
    if method == "threshold":
        return exceeds_band(current, target, cfg.rebalance.band)
    raise ValueError(f"Unknown rebalance method {method!r}")


def _validate_weights(w: pd.Series, name: str, date: pd.Timestamp) -> None:
    total = float(w.sum())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"[{name}] weights at {date.date()} sum to {total:.6f}, not 1.0")
    if (w < -1e-9).any():
        raise ValueError(f"[{name}] negative weights at {date.date()}: {w[w < 0].to_dict()}")


def run_benchmark_returns(
    returns: pd.DataFrame,
    benchmark: BenchmarkConfig,
    risk_free: str,
    warmup: int = 12,
) -> pd.Series:
    """Return the benchmark's net monthly returns over the same investable window.

    The benchmark is a static-weight portfolio rebalanced on its own schedule
    with **zero** transaction cost (it is a paper index, not a traded book).
    """
    from .config import (
        BacktestConfig,
        DataConfig,
        RebalanceConfig,
        StrategyConfig,
    )

    weights = {k: v for k, v in benchmark.weights.items() if k in returns.columns}
    if not weights:
        raise ValueError(
            f"Benchmark weights {benchmark.weights} reference no tickers in the universe {list(returns.columns)}"
        )

    bench_cfg = BacktestConfig(
        name=benchmark.name,
        tickers=list(returns.columns),
        risk_free=risk_free,
        data=DataConfig(),
        strategy=StrategyConfig(type="static_weights", params={"weights": weights}),
        rebalance=RebalanceConfig(
            method="calendar", frequency=benchmark.rebalance_frequency, cost_bps=0.0
        ),
    )
    return run_backtest(returns, bench_cfg, warmup=warmup).returns.rename(benchmark.name)
