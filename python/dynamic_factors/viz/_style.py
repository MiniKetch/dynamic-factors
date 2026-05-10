"""Shared Plotly style — same dark-quant aesthetic as vol-surface.

All viz modules import from here so the palette has one home.
"""

from __future__ import annotations


# Palette
BG_BLACK    = "#000000"
BG_PANEL    = "#0a0a0f"
GRID_FAINT  = "rgba(120,120,180,0.15)"
AXIS_FAINT  = "rgba(180,180,220,0.4)"
TEXT_BRIGHT = "rgba(230,230,255,0.95)"

# Trace colours
FACTOR_COLOR  = "#5cd8ff"   # cyan — factor 1 (market)
LOADING_POS   = "#ffd166"   # warm gold — positive loading
LOADING_NEG   = "#ff5c8a"   # hot pink  — negative loading
EQUITY_COLOR  = "#ffd166"   # gold for the equity curve
DRAWDOWN_FILL = "rgba(255,92,138,0.25)"  # pink shade for drawdown
RESID_POS     = "rgba(255,92,138,0.85)"
RESID_NEG     = "rgba(92,216,255,0.85)"


def require_plotly():
    try:
        import plotly.graph_objects as go        # noqa: WPS433
        from plotly.subplots import make_subplots  # noqa: WPS433
        return go, make_subplots
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "plotly not installed. `pip install dynamic-factors[viz]`."
        ) from exc


def axis_2d() -> dict:
    """2D axis styling — minimal grid, bright text."""
    return dict(
        gridcolor=GRID_FAINT,
        zerolinecolor=AXIS_FAINT,
        linecolor=AXIS_FAINT,
        tickfont=dict(color=TEXT_BRIGHT, size=11),
        title_font=dict(color=TEXT_BRIGHT, size=12),
    )


def base_layout(title: str) -> dict:
    """Default layout dict for a single-figure dark-mode chart."""
    return dict(
        title=dict(text=title, font=dict(color=TEXT_BRIGHT, size=18),
                   x=0.02, y=0.97),
        paper_bgcolor=BG_BLACK,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_BRIGHT),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=50, b=40),
    )
