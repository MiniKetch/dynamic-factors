"""Factor extraction layer — PCA, identification, rolling decomposition."""

from dynamic_factors.factors.pca import (
    PCAResult,
    fit_pca,
    fit_rolling_pca,
)
from dynamic_factors.factors.identification import label_factors

__all__ = [
    "PCAResult",
    "fit_pca",
    "fit_rolling_pca",
    "label_factors",
]
