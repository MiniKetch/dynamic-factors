"""Visualization layer — eigenvalues, loadings, residuals, equity curves."""

from dynamic_factors.viz.eigenvalues import render_eigenvalue_spectrum
from dynamic_factors.viz.loadings import (
    render_factor_loadings_heatmap,
    render_loading_evolution,
)
from dynamic_factors.viz.residuals import render_residual_zscore
from dynamic_factors.viz.equity import render_equity_curve

__all__ = [
    "render_eigenvalue_spectrum",
    "render_factor_loadings_heatmap",
    "render_loading_evolution",
    "render_residual_zscore",
    "render_equity_curve",
]
