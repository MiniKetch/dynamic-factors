"""Process-noise calibration for the Kalman tracker.

The single most important hyperparameter in the filter is Q, the
covariance of the per-step state drift. Wrong Q ⇒ filter either
lags reality (Q too small) or oscillates (Q too large).

We pick Q via maximum likelihood: try a small grid of candidate
values, run the filter on a representative subset of stocks, sum
the log-likelihoods, pick the Q that scored highest. This is
cheap because the filter itself is C++ — we run it 8-10 times,
not thousands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dynamic_factors._df_kernel import KalmanFilter3, KalmanFilter5


def calibrate_process_noise(
    factor_returns: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    k_factors: int = 3,
    candidates: tuple[float, ...] = (1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4),
    observation_noise: float = 1e-4,
    sample_size: int | None = 30,
    seed: int = 7,
) -> tuple[float, pd.Series]:
    """Find the Q-diagonal that maximizes the average log-likelihood.

    Parameters
    ----------
    factor_returns
        (T × k_factors) DataFrame from PCA.
    returns
        (T × N) DataFrame of stock returns.
    candidates
        Q-diagonal values to try.
    observation_noise
        Held fixed at this value during the search.
    sample_size
        How many stocks to evaluate on (random sample for speed).
        ``None`` = evaluate on all.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    (best_q, scores_series)
        ``best_q`` is the winner; ``scores_series`` maps each
        candidate to its average log-likelihood across the sample.
    """
    rng = np.random.default_rng(seed)
    if sample_size is None or sample_size >= returns.shape[1]:
        sampled = returns.columns
    else:
        sampled = rng.choice(returns.columns, size=sample_size, replace=False)

    F = factor_returns.to_numpy(dtype=float)
    cls = KalmanFilter3 if k_factors == 3 else KalmanFilter5

    scores: dict[float, float] = {}
    for q in candidates:
        Q  = q  * np.eye(k_factors)
        P0 = 1.0 * np.eye(k_factors)
        total_ll = 0.0
        for ticker in sampled:
            y = returns[ticker].dropna().to_numpy(dtype=float)
            n = min(len(y), len(F))
            if n < 30:  # skip too-short stocks
                continue
            x0 = np.zeros((k_factors, 1), dtype=float)
            kf = cls(x0, P0, Q, float(observation_noise))
            for t in range(n):
                kf.step(F[t:t + 1, :], float(y[t]))
            total_ll += float(kf.log_likelihood)
        scores[q] = total_ll / max(len(sampled), 1)

    score_series = pd.Series(scores).sort_values(ascending=False)
    best_q = float(score_series.index[0])
    return best_q, score_series
