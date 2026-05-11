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

    Implementation: precompute the boolean masks for entry/exit/NaN
    across the whole matrix, then run a single Python loop over the
    time axis that operates on whole-row numpy slices. This is
    ~20–50× faster than `iterrows()` while preserving bitwise output.
    """
    p = params or SignalParams()
    z = rolling_zscore(residuals, window=p.window, min_periods=p.min_window)
    z_arr = z.to_numpy()

    # Precomputed masks (T × N each). Compared to looping with pandas
    # per row, these run as native numpy ufuncs over the whole matrix.
    short_mask = z_arr >=  p.entry_threshold
    long_mask  = z_arr <= -p.entry_threshold
    exit_mask  = np.abs(z_arr) <= p.exit_threshold
    nan_mask   = np.isnan(z_arr)

    T, N = z_arr.shape
    out = np.zeros((T, N), dtype=np.float64)
    prev = np.zeros(N, dtype=np.float64)

    # Apply rules in the same order as the original implementation:
    #   start from prev, then set short, then long, then exit, then nan.
    # The order matters only when conditions overlap (they don't given
    # entry > exit ≥ 0), but we preserve it to keep output identical.
    for t in range(T):
        new = prev.copy()
        new[short_mask[t]] = -1.0
        new[long_mask[t]]  = +1.0
        new[exit_mask[t]]  =  0.0
        new[nan_mask[t]]   =  0.0
        out[t] = new
        prev = new

    return pd.DataFrame(out, index=residuals.index, columns=residuals.columns)
