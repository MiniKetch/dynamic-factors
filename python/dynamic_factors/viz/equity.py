"""Backtest equity-curve visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_factors.viz._style import (
    BG_BLACK, BG_PANEL, DRAWDOWN_FILL, EQUITY_COLOR,
    TEXT_BRIGHT, axis_2d, base_layout, require_plotly,
)


def render_equity_curve(
    equity: pd.Series,
    *,
    show_drawdown: bool = True,
    initial_nav: float | None = None,
    title: str = "Backtest equity curve",
    save_html: str | Path | None = None,
):
    """Equity curve with drawdown shaded below.

    Two-panel chart: top = equity (NAV) over time; bottom = drawdown
    (% from running peak), shaded pink under zero.
    """
    go, make_subplots = require_plotly()

    if initial_nav is None:
        initial_nav = float(equity.iloc[0]) if len(equity) else 1.0

    # Compute drawdown.
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0) * 100.0

    if show_drawdown:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.06,
            shared_xaxes=True,
            subplot_titles=("equity", "drawdown (%)"),
        )
    else:
        fig = make_subplots()

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", line=dict(color=EQUITY_COLOR, width=2.4),
        name="equity",
        hovertemplate="%{x|%Y-%m-%d}<br>NAV = $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    if show_drawdown:
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            mode="lines", fill="tozeroy",
            line=dict(color="rgba(255,92,138,0.9)", width=1),
            fillcolor=DRAWDOWN_FILL,
            name="drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>DD = %{y:+.2f} %<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(**base_layout(title))
    fig.update_xaxes(axis_2d(), row=1, col=1)
    fig.update_yaxes(axis_2d(), row=1, col=1, title_text="NAV ($)")
    if show_drawdown:
        fig.update_xaxes(axis_2d(), row=2, col=1, title_text="date")
        fig.update_yaxes(axis_2d(), row=2, col=1, title_text="DD (%)")

    if save_html is not None:
        path = Path(save_html)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return fig
