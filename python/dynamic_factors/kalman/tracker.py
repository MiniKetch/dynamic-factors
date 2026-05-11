"""Per-stock Kalman tracker for time-varying factor loadings.

For each stock we run a Kalman filter where the *state* is the
stock's loadings on the principal-component factors and the
*observation* is the stock's daily return. Regressors are the
factor returns at that day. Each filter step:

  state_t   = state_{t-1} + N(0, Q)              # loadings drift slowly
  obs_t     = factor_returns_t · state_t + N(0, R)

Output for each stock:
  * loadings_path: (T × k) DataFrame of filtered loadings over time
  * residual_path: (T,)     Series of innovations (= idiosyncratic
                            returns not explained by the factors)

The C++ kernel does the actual filter math; Python loops one filter
per stock and assembles the results into pandas frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from dynamic_factors._df_kernel import KalmanFilter3, KalmanFilter5


@dataclass
class LoadingPath:
    """Result of running one Kalman filter on one stock."""
    ticker: str
    loadings: pd.DataFrame   # (T × k) filtered loadings
    residuals: pd.Series     # (T,)    innovations
    # Number of filter steps where the innovation covariance went
    # non-positive — usually 0; nonzero indicates the filter became
    # numerically unstable (e.g. Q ≈ 0 with very long histories). The
    # dashboard surfaces this in the methodology tab.
    degenerate_steps: int = 0


class LoadingTracker:
    """One-stock Kalman tracker. Wraps the C++ filter with a
    pandas-friendly API."""

    def __init__(
        self,
        k_factors: int,
        *,
        initial_loading: np.ndarray | None = None,
        initial_cov_scale: float = 1.0,
        process_noise_diag: float = 1e-5,
        observation_noise: float = 1e-4,
    ):
        if k_factors not in (3, 5):
            raise ValueError(
                "k_factors must be 3 or 5 (the C++ kernel only "
                "instantiates those two sizes for performance).")
        self.k = k_factors

        if initial_loading is None:
            initial_loading = np.zeros(k_factors, dtype=float)
        if initial_loading.shape != (k_factors,):
            raise ValueError(f"initial_loading must have shape ({k_factors},)")

        Q = process_noise_diag * np.eye(k_factors, dtype=float)
        P0 = initial_cov_scale * np.eye(k_factors, dtype=float)
        x0 = initial_loading.astype(float).reshape(k_factors, 1)

        cls = KalmanFilter3 if k_factors == 3 else KalmanFilter5
        self._kf = cls(x0, P0, Q, float(observation_noise))

    def run(
        self,
        factor_returns: pd.DataFrame,
        stock_returns: pd.Series,
    ) -> LoadingPath:
        """Run the filter over an entire history.

        ``factor_returns``: (T × k) DataFrame.
        ``stock_returns``:  (T,) Series, same index.
        """
        if factor_returns.shape[1] != self.k:
            raise ValueError(
                f"factor_returns has {factor_returns.shape[1]} cols, "
                f"expected {self.k}")
        common = factor_returns.index.intersection(stock_returns.index)
        F = factor_returns.loc[common].to_numpy(dtype=float)
        y = stock_returns.loc[common].to_numpy(dtype=float)

        # Pre-allocate output arrays.
        loadings_out = np.empty((len(common), self.k), dtype=float)
        residuals_out = np.empty(len(common), dtype=float)

        # The Kalman filter's H matrix is (1 × k), so reshape each row
        # of F into a row vector for the step call.
        for t in range(len(common)):
            H = F[t:t + 1, :]
            innovation = self._kf.step(H, float(y[t]))
            residuals_out[t] = innovation
            loadings_out[t] = np.asarray(self._kf.state).flatten()

        loadings_df = pd.DataFrame(
            loadings_out, index=common,
            columns=[f"factor_{i}" for i in range(self.k)],
        )
        residuals_series = pd.Series(
            residuals_out, index=common, name=stock_returns.name or "residual",
        )
        return LoadingPath(
            ticker=str(stock_returns.name) if stock_returns.name else "?",
            loadings=loadings_df,
            residuals=residuals_series,
            degenerate_steps=int(self._kf.degenerate_steps),
        )


def track_all_stocks(
    factor_returns: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    k_factors: int = 3,
    process_noise_diag: float = 1e-5,
    observation_noise: float = 1e-4,
    progress: bool = False,
) -> dict[str, LoadingPath]:
    """Run a per-stock Kalman tracker for every column of ``returns``.

    Returns a dict keyed by ticker.
    """
    if factor_returns.shape[1] != k_factors:
        raise ValueError(
            f"factor_returns has {factor_returns.shape[1]} cols, "
            f"expected {k_factors}")

    paths: dict[str, LoadingPath] = {}
    tickers = list(returns.columns)
    for i, ticker in enumerate(tickers):
        if progress and i % 50 == 0:
            print(f"  ▸ tracker {i + 1}/{len(tickers)}: {ticker}")
        tracker = LoadingTracker(
            k_factors=k_factors,
            process_noise_diag=process_noise_diag,
            observation_noise=observation_noise,
        )
        path = tracker.run(factor_returns, returns[ticker].dropna())
        paths[ticker] = path
    return paths
