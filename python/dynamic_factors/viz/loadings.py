"""Factor-loading visuals.

Two charts:
  * **Heatmap**: factor × stock matrix. Reads at a glance whether a
    factor is "everything together" (uniform color across all
    stocks ⇒ market) or "sector tilt" (cluster of one color in one
    industry).
  * **Time-evolution**: line chart showing one stock's loading on
    one factor across the rolling-PCA windows. Tells the story of
    "betas are NOT static."
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dynamic_factors.factors.pca import PCAResult
from dynamic_factors.viz._style import (
    BG_BLACK, BG_PANEL, FACTOR_COLOR, LOADING_NEG, LOADING_POS,
    TEXT_BRIGHT, axis_2d, base_layout, require_plotly,
)


def render_factor_loadings_heatmap(
    pca: PCAResult,
    *,
    sort_by_factor: int = 0,
    sector_map: pd.Series | None = None,
    save_html: str | Path | None = None,
):
    """Heatmap of (stock × factor) loadings.

    Stocks are sorted by their loading on ``sort_by_factor`` so the
    pattern is easy to read — large positive at top, large negative
    at bottom. Optional sector_map turns the y-axis labels into
    "<sector> | <ticker>" so sector-tilt factors visibly cluster.
    """
    go, _ = require_plotly()

    loadings = pca.loadings.copy()

    if sort_by_factor < 0 or sort_by_factor >= loadings.shape[1]:
        raise ValueError(
            f"sort_by_factor out of range [0, {loadings.shape[1]})")

    sort_col = loadings.columns[sort_by_factor]
    loadings = loadings.sort_values(by=sort_col, ascending=False)

    y_labels = loadings.index.tolist()
    if sector_map is not None:
        y_labels = [
            f"{sector_map.get(t, '?')} | {t}"
            for t in loadings.index
        ]

    # Diverging colorscale: positive = gold, negative = pink.
    fig = go.Figure(go.Heatmap(
        x=loadings.columns.tolist(),
        y=y_labels,
        z=loadings.values,
        colorscale=[
            [0.0, LOADING_NEG],
            [0.5, "rgba(20,20,30,0.9)"],
            [1.0, LOADING_POS],
        ],
        zmid=0.0,
        colorbar=dict(
            title=dict(text="loading", font=dict(color=TEXT_BRIGHT)),
            tickfont=dict(color=TEXT_BRIGHT),
            outlinewidth=0,
        ),
        hovertemplate="<b>%{y}</b><br>%{x} loading = %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **base_layout(f"Factor loadings  ·  sorted by {sort_col}"),
    )
    fig.update_xaxes(side="top", **axis_2d())
    fig.update_yaxes(autorange="reversed", **axis_2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig


def render_loading_evolution(
    rolling_loadings: pd.DataFrame,
    *,
    title: str = "Factor loading evolution",
    save_html: str | Path | None = None,
):
    """Line chart of one stock's loadings on each factor over time.

    Input shape: rows = dates, cols = factor_0/factor_1/… for a
    single stock. Each factor gets its own line.
    """
    go, _ = require_plotly()

    fig = go.Figure()
    for col in rolling_loadings.columns:
        fig.add_trace(go.Scatter(
            x=rolling_loadings.index,
            y=rolling_loadings[col],
            mode="lines",
            name=col,
            line=dict(width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>"
                           "loading = %{y:.4f}<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1,
                                  dash="dot"))
    fig.update_layout(**base_layout(title))
    fig.update_xaxes(title_text="date", **axis_2d())
    fig.update_yaxes(title_text="loading", **axis_2d())

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig
