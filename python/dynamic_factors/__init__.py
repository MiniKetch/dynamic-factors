"""dynamic-factors — equity stat-arb via dynamic factor models.

Pull S&P 500 prices, decompose them into common factors via PCA,
track each stock's loading on those factors over time with a
Kalman filter, and trade the residuals as a dollar-neutral
long-short portfolio.

C++ math kernel (PCA + Kalman + rolling stats) exposed via pybind11.
Python orchestration handles data, backtest, and visualization.

Quick start:

    >>> from dynamic_factors.data import run_pipeline
    >>> result = run_pipeline()                  # offline pull + align
    >>> from dynamic_factors.factors import fit_pca
    >>> pca = fit_pca(result.returns, k_factors=3)

See ``streamlit_app.py`` for the interactive dashboard.
"""

from ._df_kernel import (
    KalmanFilter3,
    KalmanFilter5,
    RollingStats,
    fit_pca as _fit_pca_raw,
    sample_covariance,
    ledoit_wolf,
    symmetric_eigen,
)

__all__ = [
    "KalmanFilter3",
    "KalmanFilter5",
    "RollingStats",
    "sample_covariance",
    "ledoit_wolf",
    "symmetric_eigen",
]

__version__ = "0.1.0"
