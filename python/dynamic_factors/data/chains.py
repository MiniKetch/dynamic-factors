"""Batched price-history fetcher.

yfinance can pull multiple tickers in one HTTP call, but it gets
rate-limited if you send a giant batch. We chunk requests at 50
tickers each and add a small jitter between calls. For the full S&P
500 over 5 years, this completes in roughly 2-5 minutes — slow but
well within the project's offline-precompute time budget.
"""

from __future__ import annotations

import time
from typing import Iterable

import pandas as pd


def fetch_prices_batched(
    tickers: Iterable[str],
    *,
    period: str = "5y",
    interval: str = "1d",
    batch_size: int = 50,
    sleep_seconds: float = 0.4,
    auto_adjust: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Pull adjusted close prices for many tickers, batched.

    Returns a (T × N) DataFrame: rows = trading dates, columns =
    tickers, values = adjusted-close prices. Missing tickers /
    failed pulls become columns of NaN — the alignment step
    decides what to do with them.

    Parameters
    ----------
    tickers
        Iterable of ticker strings.
    period
        yfinance period string: '1y', '2y', '5y', '10y', 'max'.
    interval
        Bar size: '1d', '1wk', '1mo'.
    batch_size
        Number of tickers per yfinance call. 50 is empirically a
        sweet spot — bigger ⇒ rate-limited, smaller ⇒ overhead.
    sleep_seconds
        Pause between batches to be polite. Set to 0 if you don't
        care about rate limits.
    auto_adjust
        If True, prices are adjusted for splits & dividends. The
        right choice for return computation; the wrong choice if
        you want raw quotes.
    progress
        Print "[ batch i/N ]" lines while fetching.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance not installed. `pip install dynamic-factors[data]`."
        ) from exc

    tickers = list(tickers)
    if not tickers:
        raise ValueError("No tickers provided")

    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    frames: list[pd.DataFrame] = []

    for idx, batch in enumerate(batches, start=1):
        if progress:
            print(f"  ▸ batch {idx}/{len(batches)}: "
                  f"{len(batch)} tickers ({batch[0]} … {batch[-1]})")
        try:
            df = yf.download(
                tickers=" ".join(batch),
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
                threads=True,
                group_by="column",
            )
        except (RuntimeError, ConnectionError, ValueError) as exc:
            print(f"    ⚠ batch failed: {exc}; emitting NaN columns")
            df = pd.DataFrame()

        # yfinance returns a multi-level column DataFrame: top level is
        # the price field (Open / High / Low / Close / Volume), second
        # level is the ticker. We want the Close column for every
        # ticker, regrouped to a flat (date × ticker) DataFrame.
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                close_df = df["Close"]
            elif "Adj Close" in df.columns.get_level_values(0):
                close_df = df["Adj Close"]
            else:
                close_df = pd.DataFrame()
        elif not df.empty and "Close" in df.columns:
            # Single-ticker case — yfinance returns a flat frame.
            close_df = df[["Close"]].rename(columns={"Close": batch[0]})
        else:
            close_df = pd.DataFrame()

        # Make sure every ticker in this batch is a column, even if
        # yfinance didn't return data for it.
        for t in batch:
            if t not in close_df.columns:
                close_df[t] = float("nan")
        frames.append(close_df[batch])

        if sleep_seconds > 0 and idx < len(batches):
            time.sleep(sleep_seconds)

    # Concat along columns; pandas takes care of date-index alignment.
    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    return out.sort_index()
