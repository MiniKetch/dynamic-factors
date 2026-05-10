"""Eigenvalue spectrum — the hero chart.

The single most-recognized PCA visual: bar chart of eigenvalues
sorted descending, often with a cumulative-variance line on a
secondary axis. The 'eigenvalue cliff' (factor 1 dominates,
factor 2-3 are smaller, the rest is noise) is the headline insight
of any equity-PCA analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_factors.factors.pca import PCAResult
from dynamic_factors.viz._style import (
    AXIS_FAINT, BG_BLACK, BG_PANEL,
    FACTOR_COLOR, LOADING_POS,
    TEXT_BRIGHT, axis_2d, base_layout, require_plotly,
)


def render_eigenvalue_spectrum(
    pca: PCAResult,
    *,
    show_cumulative: bool = True,
    save_html: str | Path | None = None,
):
    """Bar chart of eigenvalues + cumulative-variance overlay.

    Each bar shows the share of total variance explained by that
    factor. The gold cumulative line is on a secondary y-axis on
    the right.
    """
    go, make_subplots = require_plotly()

    var_explained = pca.variance_explained
    cum_var       = pca.cumulative_variance_explained

    if show_cumulative:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = make_subplots()

    fig.add_trace(go.Bar(
        x=var_explained.index,
        y=var_explained.values * 100,
        name="Variance explained (%)",
        marker_color=FACTOR_COLOR,
        marker_line=dict(color="rgba(255,255,255,0.6)", width=0.5),
        hovertemplate="<b>%{x}</b><br>Variance explained = %{y:.2f} %"
                       "<extra></extra>",
    ), secondary_y=False)

    if show_cumulative:
        fig.add_trace(go.Scatter(
            x=cum_var.index,
            y=cum_var.values * 100,
            mode="lines+markers",
            name="Cumulative (%)",
            line=dict(color=LOADING_POS, width=2.5),
            marker=dict(size=8, color=LOADING_POS,
                         line=dict(color="rgba(0,0,0,0.4)", width=1)),
            hovertemplate="<b>%{x}</b><br>Cumulative = %{y:.2f} %"
                           "<extra></extra>",
        ), secondary_y=True)
        fig.update_yaxes(title_text="Cumulative variance (%)",
                          secondary_y=True,
                          range=[0, 105], **axis_2d())

    fig.update_layout(**base_layout(
        f"Eigenvalue spectrum  ·  "
        f"top {pca.k_factors} factors of {pca.n_stocks} stocks"))
    fig.update_xaxes(title_text="factor", **axis_2d())
    fig.update_yaxes(title_text="Variance explained (%)",
                      secondary_y=False, **axis_2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig
