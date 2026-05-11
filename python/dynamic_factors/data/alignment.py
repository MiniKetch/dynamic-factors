"""Time-series alignment + return computation.

Real chains have holes. yfinance occasionally returns missing rows
for individual tickers (delisted names, stale feeds, IPOs that
post-date the window). The alignment layer turns the messy pull
into a clean (T × N) panel where every column has the same date index
and the project's downstream stages can just call ``.values`` without
worrying about gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_panel(
    prices: pd.DataFrame,
    *,
    min_history_days: int = 252,
    forward_fill_limit: int = 5,
) -> pd.DataFrame:
    """Clean up a raw multi-ticker price panel.

    1. Drop tickers with fewer than ``min_history_days`` non-NaN
       observations — they're too short to be useful for a 252-day
       PCA window. This is honest survivor-bias-aware filtering.
    2. Forward-fill gaps up to ``forward_fill_limit`` business days,
       so a single missing-data day doesn't break the whole window.
    3. Drop dates where every column is still NaN (weekend / holiday
       holdovers).

    Returns a DataFrame with the same shape semantics as the input.
    """
    # Defensive: drop duplicate column names. Can happen if upstream
    # universe data has duplicate tickers, or if yfinance returns the
    # same ticker twice. Parquet writers reject duplicate columns, so
    # this guard prevents a pipeline crash several steps downstream.
    if prices.columns.duplicated().any():
        prices = prices.loc[:, ~prices.columns.duplicated(keep="first")]

    # Forward-fill short gaps.
    filled = forward_fill_limited(prices, limit=forward_fill_limit)

    # Drop tickers with insufficient history.
    valid_counts = filled.count(axis=0)
    keep = valid_counts[valid_counts >= min_history_days].index
    filled = filled[keep]

    # Drop dates that are entirely NaN.
    filled = filled.dropna(how="all")
    return filled


def forward_fill_limited(df: pd.DataFrame, *, limit: int = 5) -> pd.DataFrame:
    """Forward-fill, but at most ``limit`` consecutive NaNs.

    A bigger gap probably means the ticker truly wasn't trading; we
    don't want to fabricate prices across delistings or IPOs.
    """
    return df.ffill(limit=limit)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns ``log(P_t / P_{t-1})``.

    Log returns are additively-aggregable across time, which makes
    rolling-window operations cleaner than simple returns. The first
    row is dropped (no previous price to diff against)."""
    if len(prices) < 2:
        return prices.iloc[0:0]
    rets = np.log(prices / prices.shift(1))
    return rets.iloc[1:]
