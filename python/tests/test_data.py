"""Data layer tests — alignment + universe loading. We don't hit
yfinance from CI; the chains module's network behaviour is exercised
by the example pipeline script."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_factors.data import (
    align_panel,
    forward_fill_limited,
    load_bundled_universe,
    log_returns,
    sp500_subset,
)


def test_load_bundled_universe_has_tickers():
    u = load_bundled_universe()
    assert len(u) > 100
    assert "AAPL" in u.tickers
    assert "MSFT" in u.tickers


def test_sp500_subset_respects_n():
    u = sp500_subset(n=10)
    assert len(u) == 10


def test_log_returns_excludes_first_row():
    prices = pd.DataFrame({
        "A": [100.0, 101.0, 102.0, 103.0],
    }, index=pd.bdate_range("2024-01-01", periods=4))
    rets = log_returns(prices)
    assert len(rets) == 3
    assert rets.iloc[0]["A"] == pytest.approx(np.log(101 / 100), rel=1e-9)


def test_forward_fill_limited_caps_run_length():
    df = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0]})
    out = forward_fill_limited(df, limit=3)
    # First 1+3 entries filled, then NaN for the rest until the 7.
    assert out["A"].tolist()[:4] == [1.0, 1.0, 1.0, 1.0]
    assert pd.isna(out["A"].iloc[4])


def test_align_panel_drops_short_tickers():
    full_idx = pd.bdate_range("2024-01-01", periods=300)
    df = pd.DataFrame({
        "long":  np.arange(300, dtype=float),
        "short": [np.nan] * 295 + [1.0, 2.0, 3.0, 4.0, 5.0],
    }, index=full_idx)
    aligned = align_panel(df, min_history_days=50)
    assert "long" in aligned.columns
    assert "short" not in aligned.columns
