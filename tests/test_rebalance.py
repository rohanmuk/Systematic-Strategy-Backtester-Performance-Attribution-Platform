"""Rebalancing arithmetic: turnover, cost, drift, and schedule logic."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester import rebalance as rb


def test_turnover_and_cost_hand_computed():
    current = pd.Series({"A": 0.5, "B": 0.5})
    target = pd.Series({"A": 0.7, "B": 0.3})
    # sum|Δ| = 0.2 + 0.2 = 0.4 -> one-way turnover 0.2 ; cost = 10bps * 0.4
    one_way, cost = rb.turnover_and_cost(current, target, cost_bps=10)
    assert one_way == pytest.approx(0.2)
    assert cost == pytest.approx(10 / 10_000 * 0.4)


def test_drift_weights_conserves_and_reweights():
    w = pd.Series({"A": 0.5, "B": 0.5})
    r = pd.Series({"A": 0.10, "B": -0.10})
    new_w, port_ret = rb.drift_weights(w, r)
    assert port_ret == pytest.approx(0.0)
    assert new_w["A"] == pytest.approx(0.55)
    assert new_w["B"] == pytest.approx(0.45)
    assert new_w.sum() == pytest.approx(1.0)


def test_calendar_schedule():
    # quarterly: rebalance every 3rd period (and the first, index 0)
    flags = [rb.is_calendar_rebalance(i, "quarterly") for i in range(7)]
    assert flags == [True, False, False, True, False, False, True]
    # monthly: always
    assert all(rb.is_calendar_rebalance(i, "monthly") for i in range(5))


def test_exceeds_band():
    current = pd.Series({"A": 0.58, "B": 0.42})
    target = pd.Series({"A": 0.60, "B": 0.40})
    assert not rb.exceeds_band(current, target, band=0.05)  # 2pp drift < 5pp
    assert rb.exceeds_band(current, target, band=0.01)  # 2pp drift > 1pp
