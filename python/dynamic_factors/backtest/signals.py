"""Z-score signals from rolling residuals.

For each stock-day, we compute a z-score of the day's residual
relative to its own past N-day window. The trading rule:

  * z >= +entry  ⇒  -1 (short — residual is "too high", expect mean revert)
  * z <= -entry  ⇒  +1 (long  — residual is "too low",  expect mean revert)
  * |z| < exit   ⇒  flatten (residual is back in range)

Between thresholds (exit < |z| < entry), positions held from previous
day are kept. This hysteresis avoids whipsawing on noise around the
exit threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalParams:
    """Parameters governing signal generation."""
    window: int = 60                # rolling-window size in days
    entry_threshold: float = 2.0    # |z| ≥ this enters a position
    exit_threshold: float = 0.5     # |z| ≤ this flattens
    min_window: int = 30            # don't signal until enough data


def rolling_zscore(residuals: pd.DataFrame, window: int,
                   min_periods: int) -> pd.DataFrame:
    """Compute rolling z-score per column.

    Uses pandas' rolling() — vectorised across columns. C++ would
    be modest gain here because the pandas backend is already
    Cython-vectorised; the math is dominated by allocation, not
    arithmetic.
    """
    means = residuals.rolling(window=window, min_periods=min_periods).mean()
    stds  = residuals.rolling(window=window, min_periods=min_periods).std(ddof=1)
    # Avoid divide-by-zero. Where stdev is 0 or NaN, z is NaN.
    return (residuals - means) / stds.replace({0.0: np.nan})


def generate_signals(
    residuals: pd.DataFrame,
    params: SignalParams | None = None,
) -> pd.DataFrame:
    """Convert a (T × N) residual matrix into a {-1, 0, +1} signal
    matrix with the same shape.

    Hysteresis: a position opened on day t persists until |z|
    drops below ``exit_threshold``.
    """
    p = params or SignalParams()
    z = rolling_zscore(residuals, window=p.window, min_periods=p.min_window)

    # Positions: start with NaN, walk forward applying the rules.
    pos = pd.DataFrame(0.0, index=residuals.index, columns=residuals.columns)
    prev = pd.Series(0.0, index=residuals.columns)

    for t, row in z.iterrows():
        new = prev.copy()
        # Entry: signal sign comes from the residual's direction.
        # Big +z ⇒ short (-1); big -z ⇒ long (+1).
        new[row >=  p.entry_threshold] = -1.0
        new[row <= -p.entry_threshold] = +1.0
        # Exit: |z| within exit_threshold flattens.
        new[row.abs() <= p.exit_threshold] = 0.0
        # NaN z ⇒ no information; flatten (don't carry stale).
        new[row.isna()] = 0.0
        pos.loc[t] = new
        prev = new

    return pos
