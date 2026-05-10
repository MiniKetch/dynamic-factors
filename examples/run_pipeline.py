"""Offline pipeline: pull S&P 500 history → align → save.

Run once a week (or however fresh you want the dashboard's history).
The dashboard reads ``data/precomputed/prices.parquet`` and
``data/precomputed/returns.parquet`` produced here.

Usage:
    python examples/run_pipeline.py
    python examples/run_pipeline.py --period 2y
    python examples/run_pipeline.py --top-n 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dynamic_factors.data import (
    load_bundled_universe,
    run_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--period", default="5y",
        help="yfinance period: 1y, 2y, 5y, 10y, max. Default: 5y.",
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Limit universe to first N tickers (handy for fast iteration). "
             "Default: full bundled universe.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/precomputed"),
        help="Where to write prices.parquet + returns.parquet.",
    )
    parser.add_argument(
        "--min-history", type=int, default=252,
        help="Drop tickers with fewer than this many observations.",
    )
    args = parser.parse_args()

    universe = load_bundled_universe()
    if args.top_n is not None:
        universe = universe.head(args.top_n)

    print(f"  ▸ universe: {universe.name}  ({len(universe)} tickers)")
    result = run_pipeline(
        universe=universe,
        period=args.period,
        output_dir=args.output_dir,
        min_history_days=args.min_history,
    )
    print()
    print(f"  ✓ {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
