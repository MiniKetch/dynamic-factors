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
FACTOR_COLOR  = "#ff7c00"   # hot orange — factor 1 (peak)
LOADING_POS   = "#ff3b00"   # red-orange — positive loading (peak)
LOADING_NEG   = "#1ec79c"   # teal-green — negative loading (cool low)
EQUITY_COLOR  = "#ffc400"   # warm gold — equity curve
DRAWDOWN_FILL = "rgba(30,199,156,0.22)"  # teal-green shade for drawdown
RESID_POS     = "rgba(255,59,0,0.85)"     # warm
RESID_NEG     = "rgba(30,199,156,0.85)"   # cool

# Thermal gradient — captures the "glowing peak" aesthetic.
# Red-orange = high/hot, gold = mid, teal-green = low/cool.
# Use HEAT_DIVERGING for ±-centered data (loadings); HEAT_SEQUENTIAL for
# unsigned magnitudes (eigenvalue intensity, etc.).
HEAT_DIVERGING = [
    [0.0,  "#1ec79c"],              # negative — teal-green (cool low)
    [0.5,  "rgba(15,15,25,0.95)"],   # zero  — near-black
    [1.0,  "#ff3b00"],              # positive — orange-red (warm peak)
]
HEAT_SEQUENTIAL = [
    [0.0,  "#1ec79c"],   # teal-green
    [0.35, "#ffc400"],   # gold
    [0.7,  "#ff7c00"],   # hot orange
    [1.0,  "#ff3b00"],   # red-orange peak
]


def heat_at(t: float) -> str:
    """Sample HEAT_SEQUENTIAL at position t ∈ [0,1]. Used to colour
    discrete elements (e.g. each bar of the eigenvalue spectrum) along
    the thermal gradient."""
    stops = HEAT_SEQUENTIAL
    t = max(0.0, min(1.0, float(t)))
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i + 1]
        if x0 <= t <= x1:
            return c0 if t == x0 else (c1 if t == x1 else _lerp_color(c0, c1, (t - x0) / (x1 - x0)))
    return stops[-1][1]


def _lerp_color(c0: str, c1: str, alpha: float) -> str:
    """Linear-interpolate two hex colors. Returns rgb()."""
    def parse(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r0, g0, b0 = parse(c0)
    r1, g1, b1 = parse(c1)
    r = int(r0 + (r1 - r0) * alpha)
    g = int(g0 + (g1 - g0) * alpha)
    b = int(b0 + (b1 - b0) * alpha)
    return f"rgb({r},{g},{b})"


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
