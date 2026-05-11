"""Stock universe management.

The full S&P 500 is the project's nominal universe, but for an
honest "point-in-time" backtest you'd need a constituent history
(adds/removes by date). yfinance doesn't provide that, so we ship a
**curated snapshot** of S&P 500 large-caps and document the bias
limitation in the README — common practice for retail-grade research.

For early development we also expose ``sp500_subset(n=50)`` which
returns the n largest names by market cap order (rough). Useful when
you want to iterate fast without pulling 500 tickers every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Iterable

import pandas as pd


@dataclass
class Universe:
    """A list of tickers + optional metadata."""
    name: str
    tickers: list[str]
    description: str = ""
    metadata: pd.DataFrame | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.tickers)

    def head(self, n: int) -> "Universe":
        return Universe(
            name=f"{self.name} (top {n})",
            tickers=self.tickers[:n],
            description=self.description,
            metadata=self.metadata.head(n) if self.metadata is not None else None,
        )


def load_bundled_universe() -> Universe:
    """Load the bundled S&P 500 snapshot CSV.

    The CSV ships in ``data/snapshots/sp500.csv`` — uses
    ``importlib.resources`` so it works from a wheel install or a
    source checkout identically.
    """
    snapshot = (resources.files("dynamic_factors.data.snapshots")
                / "sp500.csv")
    with resources.as_file(snapshot) as path:
        df = pd.read_csv(path)
    # Defensive: drop any duplicate tickers in the bundled snapshot
    # so downstream code (e.g. parquet writers, which reject duplicate
    # column names) never has to think about it.
    df = df.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)
    return Universe(
        name="S&P 500 (bundled snapshot)",
        tickers=df["ticker"].tolist(),
        description="Curated S&P 500 constituents snapshot. "
                    "Static — no point-in-time accuracy.",
        metadata=df,
    )


def sp500_subset(n: int = 50) -> Universe:
    """Return the first ``n`` constituents from the bundled universe."""
    return load_bundled_universe().head(n)
