"""Offline pipeline runner: pull prices → align → save Parquet.

Designed to be invoked once a week (or however often your dashboard
needs fresh history). The dashboard then reads the Parquet outputs
without re-pulling 500 tickers on every page load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dynamic_factors.data.alignment import align_panel, log_returns
from dynamic_factors.data.chains import fetch_prices_batched
from dynamic_factors.data.universe import Universe, load_bundled_universe


# Default location for precomputed artifacts. Stays under the user's
# repo so they can inspect / version-control selectively.
DEFAULT_OUTPUT_DIR = Path("data/precomputed")


@dataclass
class PipelineResult:
    """What the pipeline produced. Returned for inspection; the
    files are also written to disk."""
    prices: pd.DataFrame
    returns: pd.DataFrame
    universe: Universe
    output_dir: Path

    def __repr__(self) -> str:
        return (f"<PipelineResult universe={self.universe.name!r} "
                f"prices=({self.prices.shape[0]}×{self.prices.shape[1]}) "
                f"returns=({self.returns.shape[0]}×{self.returns.shape[1]}) "
                f"output_dir={self.output_dir}>")


def run_pipeline(
    *,
    universe: Universe | None = None,
    period: str = "5y",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    min_history_days: int = 252,
    forward_fill_limit: int = 5,
    progress: bool = True,
) -> PipelineResult:
    """End-to-end: pull prices, align, compute log returns, save.

    Parameters
    ----------
    universe
        Universe to fetch. Defaults to the bundled S&P 500 snapshot.
    period
        yfinance period string. ``5y`` is the default — long enough
        for several PCA windows + a backtest, short enough for a
        reasonable pull.
    output_dir
        Where to save ``prices.parquet`` and ``returns.parquet``.
    min_history_days
        Drop tickers with fewer non-NaN observations.
    forward_fill_limit
        Max consecutive NaNs to forward-fill.
    progress
        Print step-by-step progress lines.
    """
    if universe is None:
        universe = load_bundled_universe()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        print(f"  ▸ pulling {len(universe)} tickers, period={period}…")
    raw = fetch_prices_batched(
        universe.tickers, period=period, progress=progress,
    )

    if progress:
        print(f"  ▸ raw panel: {raw.shape[0]} dates × {raw.shape[1]} tickers")
        print("  ▸ aligning + filtering…")
    aligned = align_panel(
        raw,
        min_history_days=min_history_days,
        forward_fill_limit=forward_fill_limit,
    )
    rets = log_returns(aligned)

    # Filter the universe metadata down to whatever survived.
    surviving_tickers = aligned.columns.tolist()
    if universe.metadata is not None:
        meta = universe.metadata[
            universe.metadata["ticker"].isin(surviving_tickers)
        ].reset_index(drop=True)
    else:
        meta = None
    surviving_universe = Universe(
        name=universe.name + " (post-alignment)",
        tickers=surviving_tickers,
        description=universe.description,
        metadata=meta,
    )

    if progress:
        print(f"  ▸ aligned panel: {aligned.shape[0]} dates × "
              f"{aligned.shape[1]} tickers ({len(universe) - aligned.shape[1]} dropped)")

    # Save Parquet snapshots. .ffill metadata drops in pyarrow but the
    # important bit (the panel) survives cleanly.
    prices_path  = out_dir / "prices.parquet"
    returns_path = out_dir / "returns.parquet"
    aligned.to_parquet(prices_path)
    rets.to_parquet(returns_path)
    if progress:
        print(f"  ✓ wrote {prices_path}")
        print(f"  ✓ wrote {returns_path}")

    return PipelineResult(
        prices=aligned,
        returns=rets,
        universe=surviving_universe,
        output_dir=out_dir,
    )
