"""Market-data acquisition, caching, and monthly total-return construction.

Design goals:

* **Reproducible & offline.** Raw daily adjusted-close panels are cached to
  ``data/*.parquet`` keyed by the ticker set. After the first successful fetch
  the project runs with no network access; ``force_refresh=True`` updates it.
* **Total return.** ``yfinance(auto_adjust=True)`` returns split- and
  dividend-adjusted closes, so month-over-month percentage changes are total
  returns (price + reinvested income), which is what we backtest.
* **Explicit missing-data handling.** We resample to month-end, then trim to
  the *longest common history* (first month in which every ticker has data).
  Any tickers dropped for insufficient history, and any internal gaps that are
  forward-filled, are logged rather than silently swallowed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

#: On-disk cache location (git-ignored). Sits next to the package's repo root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PriceData:
    """Resolved data for a backtest: aligned monthly returns plus provenance."""

    monthly_returns: pd.DataFrame  # index = month-end, columns = tickers
    daily_prices: pd.DataFrame  # raw adjusted closes (for reference/plots)
    start: pd.Timestamp
    end: pd.Timestamp
    dropped_tickers: List[str]


def _cache_key(tickers: List[str]) -> str:
    digest = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:10]
    return f"prices_{digest}"


def _read_cache(key: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / f"{key}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # pragma: no cover - corrupt/unreadable cache
            logger.warning("Could not read parquet cache at %s; will re-download.", path)
    return None


def _write_cache(key: str, prices: pd.DataFrame, tickers: List[str]) -> None:
    prices.to_parquet(DATA_DIR / f"{key}.parquet")
    meta = {
        "tickers": sorted(tickers),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(prices)),
    }
    (DATA_DIR / f"{key}.meta.json").write_text(json.dumps(meta, indent=2))


def download_prices(
    tickers: List[str],
    start: str = "2000-01-01",
    end: Optional[str] = None,
    cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily adjusted-close prices, one column per ticker.

    Uses the on-disk cache when it already contains every requested ticker
    (offline-friendly); otherwise downloads the full set via yfinance and
    refreshes the cache. Never fabricates data — a ticker that yfinance cannot
    return is simply absent from the result (and logged).
    """
    tickers = sorted(set(tickers))
    key = _cache_key(tickers)

    if cache and not force_refresh:
        cached = _read_cache(key)
        if cached is not None and set(tickers).issubset(cached.columns):
            logger.info("Using cached daily prices for %d tickers.", len(tickers))
            return cached[tickers]

    end = end or dt.date.today().isoformat()
    logger.info("Downloading %d tickers from yfinance (%s -> %s).", len(tickers), start, end)

    import yfinance as yf  # imported lazily so offline cache hits need no network stack

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # 'Close' becomes the total-return-adjusted price
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {tickers}. Check symbols and connectivity."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame(
            {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)}
        )
    else:  # single-ticker request collapses the column MultiIndex
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index().dropna(how="all")
    missing = set(tickers) - set(prices.columns)
    if missing:
        logger.warning("yfinance returned no data for: %s", sorted(missing))

    if cache:
        _write_cache(key, prices, list(prices.columns))
    return prices


def to_monthly_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily adjusted closes to monthly total returns.

    We take the last observed price in each calendar month (month-end sampling)
    and compound to a simple monthly return via percentage change. The first
    month becomes NaN (no prior anchor) and is dropped. If the most recent month
    is still in progress (the latest price is well before its month-end), that
    final partial month is dropped so every reported return covers a full month.
    """
    monthly_prices = daily_prices.resample("ME").last()

    last_price_date = daily_prices.dropna(how="all").index.max()
    last_label = monthly_prices.index.max()
    # A "complete" month has its last observation within a few trading days of
    # month-end; otherwise the month is still running -> drop it.
    if (last_label - last_price_date).days > 5:
        logger.info("Dropping incomplete final month %s (last price %s).", last_label.date(), last_price_date.date())
        monthly_prices = monthly_prices.iloc[:-1]

    return monthly_prices.pct_change().dropna(how="all")


def align_common_history(
    monthly_returns: pd.DataFrame,
    min_months: int = 24,
) -> tuple[pd.DataFrame, List[str]]:
    """Trim to the longest window where *every* remaining ticker has data.

    Tickers whose history is shorter than ``min_months`` after the common start
    would force the whole panel to be truncated to almost nothing; such tickers
    are dropped (and returned) rather than allowed to dominate the window.
    Internal gaps (holidays, listing mismatches) are forward-filled with 0
    return only *within* each column's live range — never before its inception.
    """
    df = monthly_returns.copy()

    # Drop tickers whose *own* history is shorter than min_months first — a
    # late-listing ETF would otherwise collapse the common window for everyone.
    counts = df.notna().sum()
    too_short = counts[counts < min_months].index.tolist()
    if too_short:
        logger.warning(
            "Dropping tickers with < %d months of history: %s", min_months, too_short
        )
        df = df.drop(columns=too_short)

    # Longest window where every surviving ticker has data.
    first_valid = df.apply(lambda col: col.first_valid_index())
    common_start = first_valid.max()

    aligned = df.loc[common_start:].copy()
    # Fill any sporadic interior gaps with 0 (no observed move) and log them.
    interior_gaps = int(aligned.isna().sum().sum())
    if interior_gaps:
        logger.warning("Filling %d interior monthly gaps with 0 return.", interior_gaps)
        aligned = aligned.fillna(0.0)

    return aligned, too_short


def load_price_data(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache: bool = True,
    force_refresh: bool = False,
) -> PriceData:
    """End-to-end: download → monthly returns → common-history alignment.

    ``start=None`` requests the longest available history (we ask yfinance from
    2000 and let alignment find the true common start). ``end=None`` means today.
    """
    fetch_start = start or "2000-01-01"
    daily = download_prices(tickers, start=fetch_start, end=end, cache=cache, force_refresh=force_refresh)
    monthly = to_monthly_returns(daily)

    if start is not None:
        monthly = monthly.loc[pd.Timestamp(start):]

    aligned, dropped = align_common_history(monthly)
    logger.info(
        "Aligned universe: %d assets, %s -> %s (%d months).",
        aligned.shape[1],
        aligned.index.min().date(),
        aligned.index.max().date(),
        len(aligned),
    )
    return PriceData(
        monthly_returns=aligned,
        daily_prices=daily,
        start=aligned.index.min(),
        end=aligned.index.max(),
        dropped_tickers=dropped,
    )
