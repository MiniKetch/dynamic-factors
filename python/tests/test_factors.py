"""PCA wrapper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_factors.factors import fit_pca, fit_rolling_pca, label_factors


@pytest.fixture
def synthetic_returns():
    """Single-factor model: every stock loads on one common factor."""
    rng = np.random.default_rng(42)
    T, N = 252, 20
    market = rng.standard_normal(T)
    betas = rng.uniform(0.6, 1.4, size=N)
    rets = np.outer(market, betas) + 0.05 * rng.standard_normal((T, N))
    return pd.DataFrame(
        rets,
        index=pd.bdate_range("2024-01-01", periods=T),
        columns=[f"S{i:02d}" for i in range(N)],
    )


def test_fit_pca_shape(synthetic_returns):
    pca = fit_pca(synthetic_returns, k_factors=3)
    assert pca.eigenvalues.shape == (3,)
    assert pca.loadings.shape == (synthetic_returns.shape[1], 3)
    assert pca.factor_returns.shape == (synthetic_returns.shape[0], 3)


def test_fit_pca_first_factor_dominant(synthetic_returns):
    """One common factor in the data ⇒ factor 1 explains > 80% of variance."""
    pca = fit_pca(synthetic_returns, k_factors=3)
    assert pca.variance_explained.iloc[0] > 0.80


def test_fit_pca_rejects_nan():
    df = pd.DataFrame({"A": [1.0, np.nan], "B": [2.0, 3.0]})
    with pytest.raises(ValueError, match="NaN"):
        fit_pca(df, k_factors=1)


def test_fit_pca_rejects_invalid_k(synthetic_returns):
    with pytest.raises(ValueError):
        fit_pca(synthetic_returns, k_factors=0)
    with pytest.raises(ValueError):
        fit_pca(synthetic_returns, k_factors=999)


def test_variance_explained_sums_to_total(synthetic_returns):
    pca = fit_pca(synthetic_returns, k_factors=synthetic_returns.shape[1])
    # When k = N, the kept eigenvalues sum to total variance.
    assert pca.eigenvalues.sum() == pytest.approx(pca.total_variance, rel=1e-9)


def test_rolling_pca_yields_results(synthetic_returns):
    out = list(fit_rolling_pca(
        synthetic_returns, k_factors=3, window=120, step=30,
    ))
    assert len(out) >= 4
    for date, result in out:
        assert isinstance(date, pd.Timestamp)
        assert result.k_factors == 3


def test_rolling_pca_sign_alignment(synthetic_returns):
    """Sign-aligned loadings should never have a 180° flip between
    consecutive windows on stable data — check that the first
    factor's loading on stock 0 keeps the same sign throughout."""
    out = list(fit_rolling_pca(
        synthetic_returns, k_factors=3, window=120, step=30,
        align_signs_across_windows=True,
    ))
    signs = [np.sign(result.loadings.iloc[0, 0]) for _, result in out]
    # All same sign.
    assert len(set(signs)) == 1


def test_label_factors_detects_market(synthetic_returns):
    pca = fit_pca(synthetic_returns, k_factors=3)
    labels = label_factors(pca)
    # First factor (eigenvalue largest, all-positive loadings) should
    # land on "market".
    assert labels.iloc[0] == "market"
