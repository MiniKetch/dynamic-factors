"""Pythonic PCA wrapper around the C++ kernel.

The C++ kernel exposes ``fit_pca(returns_matrix, k, shrinkage)`` —
this layer adds dataframe-aware ergonomics (named index/columns
preserved through the decomposition) and handles the rolling-window
case where signs need to be aligned across consecutive fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from dynamic_factors._df_kernel import fit_pca as _fit_pca_raw


@dataclass
class PCAResult:
    """Tickers / dates preserved as labels around the C++ output."""
    eigenvalues: pd.Series          # index = factor_0, factor_1, …
    loadings: pd.DataFrame          # rows = stocks, cols = factor_i
    factor_returns: pd.DataFrame    # rows = dates, cols = factor_i
    total_variance: float
    k_factors: int

    @property
    def variance_explained(self) -> pd.Series:
        """Each factor's share of the total variance."""
        return self.eigenvalues / self.total_variance

    @property
    def cumulative_variance_explained(self) -> pd.Series:
        return self.variance_explained.cumsum()

    @property
    def n_stocks(self) -> int:
        return self.loadings.shape[0]

    @property
    def n_dates(self) -> int:
        return self.factor_returns.shape[0]


def fit_pca(
    returns: pd.DataFrame,
    k_factors: int = 3,
    *,
    shrinkage: bool = True,
) -> PCAResult:
    """Fit PCA on a returns DataFrame and keep the top ``k_factors``.

    Parameters
    ----------
    returns
        (T × N) DataFrame; rows = dates, cols = tickers, values =
        log returns. Must have no NaN values — call
        ``align_panel(...)`` first if it might.
    k_factors
        Number of components to keep.
    shrinkage
        Use Ledoit-Wolf shrunk covariance instead of raw sample
        covariance. Recommended whenever T isn't much larger than N.
    """
    if returns.isna().any().any():
        raise ValueError(
            "fit_pca: input DataFrame contains NaN — align/clean first.")
    if k_factors < 1 or k_factors > returns.shape[1]:
        raise ValueError(
            f"k_factors must be in [1, {returns.shape[1]}], got {k_factors}")

    raw = _fit_pca_raw(returns.to_numpy(dtype=float),
                        k_factors=int(k_factors),
                        shrinkage=bool(shrinkage))

    factor_names = [f"factor_{i}" for i in range(k_factors)]
    return PCAResult(
        eigenvalues=pd.Series(raw["eigenvalues"], index=factor_names,
                               name="eigenvalue"),
        loadings=pd.DataFrame(raw["loadings"], index=returns.columns,
                               columns=factor_names),
        factor_returns=pd.DataFrame(raw["factor_returns"],
                                     index=returns.index,
                                     columns=factor_names),
        total_variance=float(raw["total_variance"]),
        k_factors=int(k_factors),
    )


def fit_rolling_pca(
    returns: pd.DataFrame,
    *,
    k_factors: int = 3,
    window: int = 252,
    step: int = 5,
    shrinkage: bool = True,
    align_signs_across_windows: bool = True,
) -> Iterator[tuple[pd.Timestamp, PCAResult]]:
    """Yield (window_end_date, PCAResult) for each rolling fit.

    The window is right-aligned: at date T, the fit uses the trailing
    ``window`` rows up to and including T. Step controls how often
    we re-fit (``step=5`` = weekly).

    Sign alignment: PCA eigenvectors are arbitrary up to sign. Without
    correction, the loading of any given factor can flip between
    consecutive fits, making the time-evolution look like noise.
    The aligner compares each new loading vector to the previous fit
    and flips the sign if the dot product is negative.
    """
    if window < 30:
        raise ValueError("window must be at least 30 (a meaningful covariance)")
    if step < 1:
        raise ValueError("step must be ≥ 1")

    n_rows = len(returns)
    if n_rows < window:
        return

    previous: PCAResult | None = None
    for end in range(window, n_rows + 1, step):
        slice_df = returns.iloc[end - window:end]
        result = fit_pca(slice_df, k_factors=k_factors, shrinkage=shrinkage)

        # Sign-align loadings vs the previous window.
        if previous is not None and align_signs_across_windows:
            for i in range(k_factors):
                col = result.loadings.columns[i]
                dot = float(result.loadings[col].dot(previous.loadings[col]))
                if dot < 0.0:
                    result.loadings[col] = -result.loadings[col]
                    result.factor_returns[col] = -result.factor_returns[col]

        previous = result
        yield slice_df.index[-1], result
