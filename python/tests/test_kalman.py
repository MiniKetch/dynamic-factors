"""Kalman tracker tests — exercise the C++ filter through the
Python wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_factors.kalman import LoadingTracker, calibrate_process_noise


@pytest.fixture
def synthetic_factors_and_stock():
    rng = np.random.default_rng(7)
    T = 252
    F = pd.DataFrame(
        rng.standard_normal((T, 3)) * 0.01,
        index=pd.bdate_range("2024-01-01", periods=T),
        columns=["factor_0", "factor_1", "factor_2"],
    )
    true_beta = np.array([0.8, 0.3, -0.2])
    y = F.values @ true_beta + 0.005 * rng.standard_normal(T)
    return F, pd.Series(y, index=F.index, name="STK"), true_beta


def test_tracker_recovers_static_betas(synthetic_factors_and_stock):
    """When loadings are constant, the filter should converge to them."""
    factors, ret, true_beta = synthetic_factors_and_stock
    tracker = LoadingTracker(
        k_factors=3,
        process_noise_diag=1e-6,
        observation_noise=1e-4,
    )
    path = tracker.run(factors, ret)
    final = path.loadings.iloc[-1].to_numpy()
    np.testing.assert_allclose(final, true_beta, atol=0.05)


def test_tracker_residuals_have_zero_mean(synthetic_factors_and_stock):
    """Innovation series should have ~zero mean if the model is correct."""
    factors, ret, _ = synthetic_factors_and_stock
    tracker = LoadingTracker(k_factors=3, process_noise_diag=1e-6)
    path = tracker.run(factors, ret)
    # Skip the warm-up period.
    np.testing.assert_allclose(path.residuals.iloc[60:].mean(), 0.0, atol=0.005)


def test_tracker_rejects_wrong_k():
    factors = pd.DataFrame(np.random.randn(10, 5))
    ret = pd.Series(np.random.randn(10))
    with pytest.raises(ValueError, match="k_factors"):
        LoadingTracker(k_factors=4)


def test_tracker_rejects_dimension_mismatch():
    factors = pd.DataFrame(np.random.randn(10, 3))
    ret = pd.Series(np.random.randn(10))
    tracker = LoadingTracker(k_factors=5)
    with pytest.raises(ValueError, match="cols"):
        tracker.run(factors, ret)


def test_calibrate_process_noise_picks_finite_q(synthetic_factors_and_stock):
    factors, ret, _ = synthetic_factors_and_stock
    returns_df = pd.DataFrame({"STK": ret})
    best_q, scores = calibrate_process_noise(
        factors, returns_df,
        candidates=(1e-7, 1e-5, 1e-3),
        sample_size=None,
    )
    assert best_q in (1e-7, 1e-5, 1e-3)
    assert len(scores) == 3
    # All scores should be finite (no NaN log-likelihoods).
    assert np.all(np.isfinite(scores.values))
