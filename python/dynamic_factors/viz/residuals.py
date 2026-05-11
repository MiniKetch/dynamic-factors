"""Residual time-series + z-score visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_factors.viz._style import (
    BG_BLACK, BG_PANEL, FACTOR_COLOR, LOADING_NEG, LOADING_POS,
    RESID_NEG, RESID_POS, TEXT_BRIGHT, axis_2d, base_layout, require_plotly,
)


def render_residual_zscore(
    residual: pd.Series,
    zscore: pd.Series,
    *,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    title: str | None = None,
    save_html: str | Path | None = None,
):
    """Two-panel chart: raw residual on top, rolling z-score below
    with entry/exit threshold lines drawn in.

    The z-score panel is the trading view — bars colored by sign,
    horizontal lines at ±entry and ±exit show the strategy's
    decision zones.
    """
    go, make_subplots = require_plotly()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.45, 0.55],
        vertical_spacing=0.06,
        shared_xaxes=True,
        subplot_titles=("residual", "z-score (rolling)"),
    )

    # --- Top: residual time series ---
    fig.add_trace(go.Scatter(
        x=residual.index, y=residual.values,
        mode="lines", line=dict(color=FACTOR_COLOR, width=1.4),
        name="residual",
        hovertemplate="%{x|%Y-%m-%d}<br>residual = %{y:+.5f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1),
                   row=1, col=1)

    # --- Bottom: z-score with thresholds ---
    bar_colors = [RESID_POS if v > 0 else RESID_NEG for v in zscore.values]
    fig.add_trace(go.Bar(
        x=zscore.index, y=zscore.values,
        marker_color=bar_colors,
        name="z-score",
        hovertemplate="%{x|%Y-%m-%d}<br>z = %{y:+.2f}<extra></extra>",
    ), row=2, col=1)
    # Each tuple is (y_level, label, is_entry). is_entry is an explicit
    # flag so the styling logic doesn't have to parse the label string.
    for level, _label, is_entry in [
        ( entry_threshold, "+entry", True),
        (-entry_threshold, "−entry", True),
        ( exit_threshold,  "+exit",  False),
        (-exit_threshold,  "−exit",  False),
    ]:
        fig.add_hline(
            y=level,
            line=dict(
                color=LOADING_POS if is_entry else "rgba(255,255,255,0.3)",
                width=1, dash="dash" if is_entry else "dot",
            ),
            row=2, col=1,
        )
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.4)", width=1),
                   row=2, col=1)

    fig.update_layout(**base_layout(title or "Residual & z-score"))
    fig.update_xaxes(axis_2d(), row=2, col=1, title_text="date")
    fig.update_xaxes(axis_2d(), row=1, col=1)
    fig.update_yaxes(axis_2d(), row=1, col=1)
    fig.update_yaxes(axis_2d(), row=2, col=1, title_text="z")

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig
