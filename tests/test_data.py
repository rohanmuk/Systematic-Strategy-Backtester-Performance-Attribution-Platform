"""Data pipeline: monthly resampling, common-history alignment, gap handling.

All offline — synthetic price/return panels, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import data as dm


def test_to_monthly_returns_uses_month_end_last():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    prices = pd.Series(100.0, index=idx, name="A")
    # Force specific month-end levels: Jan=110, Feb=121, Mar=133.1 (each +10%).
    prices.loc[idx[idx.month == 1][-1]] = 110.0
    prices.loc[idx[idx.month == 2][-1]] = 121.0
    prices.loc[idx[idx.month == 3][-1]] = 133.1
    monthly = dm.to_monthly_returns(prices.to_frame())
    assert list(monthly.index.month) == [2, 3]
    assert monthly["A"].tolist() == pytest.approx([0.10, 0.10])


def test_incomplete_final_month_dropped():
    # Data stops on the 10th -> the final (partial) month must be dropped.
    idx = pd.bdate_range("2020-01-01", "2020-03-10")
    prices = pd.DataFrame({"A": np.linspace(100, 110, len(idx))}, index=idx)
    monthly = dm.to_monthly_returns(prices)
    assert monthly.index.max().month == 2  # March dropped


def test_align_common_history_drops_short_and_fills_gaps():
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "EARLY": rng.normal(0.005, 0.02, 60),
            "MID": rng.normal(0.004, 0.02, 60),
            "LATE": rng.normal(0.004, 0.02, 60),
        },
        index=idx,
    )
    df.loc[idx[:12], "MID"] = np.nan  # MID starts at month 12 (48 obs -> kept)
    df.loc[idx[:40], "LATE"] = np.nan  # LATE starts at month 40 (20 obs -> dropped)
    df.loc[idx[20], "EARLY"] = np.nan  # a single interior gap

    aligned, dropped = dm.align_common_history(df, min_months=24)

    assert "LATE" in dropped and "LATE" not in aligned.columns
    assert aligned.index.min() == idx[12]  # common start = MID's inception
    assert not aligned.isna().any().any()  # interior gap filled
    assert aligned.loc[idx[20], "EARLY"] == 0.0  # filled with zero return
