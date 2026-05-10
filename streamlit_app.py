"""dynamic-factors — interactive dashboard.

Run:
    pip install -e ".[data,viz,dashboard]"
    streamlit run streamlit_app.py

Architecture (hybrid mode):
    * History (multi-year prices, returns, factor decompositions,
      Kalman residuals, backtest equity) is **precomputed** by
      ``examples/run_pipeline.py`` and stored in ``data/precomputed/``.
    * The dashboard reads those parquet files for the historical
      tabs; the "Today" tab refreshes live from yfinance.

If you haven't run the pipeline yet, the dashboard surfaces a clear
prompt with the command. No silent NaN charts.
"""

from __future__ import annotations

import dataclasses
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dynamic_factors.backtest import (
    BacktestConfig, run_backtest, SignalParams,
)
from dynamic_factors.factors import fit_pca, fit_rolling_pca, label_factors
from dynamic_factors.kalman import LoadingTracker, track_all_stocks
from dynamic_factors.viz import (
    render_eigenvalue_spectrum,
    render_equity_curve,
    render_factor_loadings_heatmap,
    render_loading_evolution,
    render_residual_zscore,
)


# ============================================================================
# Page setup
# ============================================================================

st.set_page_config(
    page_title="dynamic-factors",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 28px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem; padding: 0.6rem 1.2rem;
    }
    [data-testid="stMetricValue"] { font-size: 2.0rem; }
    [data-testid="stMetricLabel"] { font-size: 0.95rem; opacity: 0.75; }
    .vsc-tall .js-plotly-plot { min-height: 700px; }
    hr { border-color: rgba(255,255,255,0.08); }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌌 dynamic-factors")
st.markdown(
    "<p style='font-size:1.1rem; opacity:0.8; margin-top:-0.5rem;'>"
    "PCA on the S&P 500 + Kalman-filtered factor loadings + "
    "stat-arb backtest. C++ kernel · pybind11 · Streamlit."
    "</p>",
    unsafe_allow_html=True,
)


# ============================================================================
# Cache layer — same attrs-survival pattern as vol-surface.
# ============================================================================

DATA_DIR = Path("data/precomputed")


@st.cache_data(ttl=600, show_spinner="Loading precomputed prices…")
def load_prices() -> pd.DataFrame | None:
    p = DATA_DIR / "prices.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data(ttl=600, show_spinner="Loading precomputed returns…")
def load_returns() -> pd.DataFrame | None:
    r = DATA_DIR / "returns.parquet"
    if not r.exists():
        return None
    return pd.read_parquet(r)


@st.cache_data(ttl=600, show_spinner="Fitting PCA…")
def cached_pca(_returns_hash: tuple, k: int):
    """Cache key uses the returns hash; the underlying frame is
    passed via session state to avoid recomputing on every rerun."""
    returns = st.session_state["returns"]
    pca = fit_pca(returns, k_factors=k)
    return {
        "eigenvalues": pca.eigenvalues,
        "loadings": pca.loadings,
        "factor_returns": pca.factor_returns,
        "total_variance": pca.total_variance,
        "k_factors": pca.k_factors,
    }


@st.cache_data(ttl=600, show_spinner="Tracking time-varying loadings…")
def cached_kalman_residuals(_pca_hash: tuple, _q: float):
    """Run the Kalman tracker for every stock and assemble the
    residual matrix. Cached because it's the most expensive step."""
    factor_returns = st.session_state["factor_returns"]
    returns = st.session_state["returns"]
    paths = track_all_stocks(
        factor_returns, returns,
        k_factors=factor_returns.shape[1],
        process_noise_diag=_q,
    )
    residuals = pd.DataFrame(
        {t: paths[t].residuals for t in paths}
    ).reindex(returns.index)
    # Also save loadings paths for the time-evolution chart (per ticker).
    return residuals, {t: paths[t].loadings for t in paths}


@st.cache_data(ttl=600, show_spinner="Running backtest…")
def cached_backtest(_residuals_hash: tuple, _q: float, entry: float,
                     window: int):
    residuals = st.session_state["residuals"]
    prices    = st.session_state["prices"]
    cfg = BacktestConfig(
        signal_params=SignalParams(
            window=window,
            entry_threshold=entry,
            exit_threshold=0.5,
        ),
    )
    result = run_backtest(residuals, prices, config=cfg)
    return {
        "equity": result.equity,
        "pnl": result.pnl,
        "turnover": result.turnover,
        "costs": result.costs,
        "total_return": result.total_return,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
    }


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("Model")
    k_factors = st.slider("Number of factors", 2, 5, 3,
                           help="Top eigenvectors kept after PCA. 3 is the "
                                "classic 'market + 2 sectors' decomposition.")
    process_noise = st.select_slider(
        "Kalman process noise (Q diag)",
        options=[1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
        value=1e-5,
        format_func=lambda v: f"{v:.0e}",
        help="Bigger ⇒ loadings drift faster. Smaller ⇒ filter lags reality.",
    )

    st.divider()
    st.header("Backtest")
    entry_threshold = st.slider(
        "Entry threshold (|z|)", 1.0, 3.0, 2.0, step=0.1,
        help="Open a position when residual z-score exceeds this.",
    )
    zscore_window = st.slider(
        "Z-score rolling window (days)", 20, 120, 60,
    )

    st.divider()
    st.caption("History pre-computed by `examples/run_pipeline.py`. "
                "Rerun the pipeline to refresh.")


# ============================================================================
# Data load
# ============================================================================

prices = load_prices()
returns = load_returns()

if prices is None or returns is None:
    st.error(
        "**No precomputed data found.**  "
        "Run the offline pipeline first to populate `data/precomputed/`:"
    )
    st.code(
        "python examples/run_pipeline.py",
        language="bash",
    )
    st.info(
        "The pipeline pulls ~5 years of S&P 500 daily closes from yfinance "
        "(~5-15 minutes the first time) and saves clean Parquet files to "
        "`data/precomputed/`. Subsequent dashboard loads are instant."
    )
    st.stop()

st.session_state["prices"] = prices
st.session_state["returns"] = returns

# Top metrics.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers", f"{returns.shape[1]:,}")
c2.metric("Trading days", f"{returns.shape[0]:,}")
c3.metric("Date range", f"{returns.index[0].date()}")
c4.metric("Through", f"{returns.index[-1].date()}")

st.write("")


# ============================================================================
# Run model
# ============================================================================

# Cheap hashes that change when the underlying data does.
returns_hash = (returns.shape, str(returns.index[0]), str(returns.index[-1]))

pca_dict = cached_pca(returns_hash, k_factors)
factor_returns = pca_dict["factor_returns"]
st.session_state["factor_returns"] = factor_returns

# Reconstruct PCAResult from the cached dict (cache_data needs primitives).
from dynamic_factors.factors.pca import PCAResult
pca = PCAResult(
    eigenvalues=pca_dict["eigenvalues"],
    loadings=pca_dict["loadings"],
    factor_returns=factor_returns,
    total_variance=pca_dict["total_variance"],
    k_factors=pca_dict["k_factors"],
)

residuals, loading_paths = cached_kalman_residuals(returns_hash, process_noise)
st.session_state["residuals"] = residuals

bt_dict = cached_backtest(returns_hash, process_noise,
                           entry_threshold, zscore_window)


# ============================================================================
# Tabs
# ============================================================================

tabs = st.tabs([
    "📊  Eigenvalue spectrum",
    "🌡️  Factor loadings",
    "🌀  Loading evolution",
    "🎯  Residual & z-score",
    "💰  Backtest equity",
    "📋  Methodology",
])
tab_eig, tab_load, tab_evo, tab_resid, tab_eq, tab_about = tabs


def _snapshot_button(fig, *, key: str, default_name: str):
    try:
        png_bytes = fig.to_image(format="png", width=1600, height=1000, scale=2)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"📷 PNG snapshot unavailable ({exc.__class__.__name__}). "
                   "Install kaleido: `pip install kaleido`.")
        return
    st.download_button(
        "📷  Snapshot for socials (PNG)",
        data=png_bytes, file_name=default_name,
        mime="image/png", key=key,
    )


# --- Eigenvalue spectrum ---
with tab_eig:
    fig = render_eigenvalue_spectrum(pca)
    st.plotly_chart(fig, width="stretch", height=560, key="eig_chart")
    st.caption(
        "Bars: each factor's share of total variance. Gold line: "
        "cumulative variance explained as you add factors. The "
        "'eigenvalue cliff' (factor 1 dominates, the rest are smaller) "
        "is the canonical equity-PCA signature."
    )
    _snapshot_button(fig, key="snap_eig", default_name="eigenvalues.png")


# --- Factor loadings heatmap ---
with tab_load:
    sort_factor = st.selectbox(
        "Sort stocks by loading on",
        list(range(k_factors)),
        format_func=lambda i: f"factor_{i}",
    )
    # Identify factors via simple labelling.
    labels = label_factors(pca)
    if not labels.empty:
        st.caption(
            "Auto-detected factor labels: "
            + " · ".join(f"`{k}` → **{v}**" for k, v in labels.items())
        )

    fig = render_factor_loadings_heatmap(pca, sort_by_factor=sort_factor)
    st.plotly_chart(fig, width="stretch", height=720, key="load_chart")
    st.caption(
        "Stocks (rows) × factors (cols). Gold = positive loading, "
        "pink = negative. Sort by any factor to see the cluster "
        "structure cleanly."
    )
    _snapshot_button(fig, key="snap_load", default_name="loadings.png")


# --- Loading evolution ---
with tab_evo:
    ticker_choice = st.selectbox(
        "Stock", sorted(loading_paths.keys()), index=0,
    )
    fig = render_loading_evolution(
        loading_paths[ticker_choice],
        title=f"{ticker_choice} factor loadings (Kalman-filtered)",
    )
    st.plotly_chart(fig, width="stretch", height=560, key="evo_chart")
    st.caption(
        "Each line: how this stock's loading on a given factor "
        "evolves over time. Watch how betas rotate around regime "
        "shifts (Covid, rate-hike cycles, AI rally) — they're "
        "anything but static."
    )
    _snapshot_button(fig, key="snap_evo",
                      default_name=f"{ticker_choice}_evolution.png")


# --- Residual & z-score ---
with tab_resid:
    ticker_choice = st.selectbox(
        "Stock", sorted(residuals.columns.tolist()),
        index=0, key="resid_ticker",
    )
    resid = residuals[ticker_choice].dropna()
    rolling_mean = resid.rolling(zscore_window, min_periods=20).mean()
    rolling_std  = resid.rolling(zscore_window, min_periods=20).std(ddof=1)
    z = (resid - rolling_mean) / rolling_std.replace({0.0: np.nan})

    fig = render_residual_zscore(
        resid, z,
        entry_threshold=entry_threshold,
        title=f"{ticker_choice} residual",
    )
    st.plotly_chart(fig, width="stretch", height=560, key="resid_chart")
    st.caption(
        "Top: idiosyncratic return after stripping factor exposure. "
        "Bottom: rolling z-score with entry/exit decision zones. "
        "The strategy enters short when z exceeds +2, long when z "
        "drops below -2, and flattens when |z| < 0.5."
    )
    _snapshot_button(fig, key="snap_resid",
                      default_name=f"{ticker_choice}_residual.png")


# --- Backtest equity curve ---
with tab_eq:
    cols = st.columns(4)
    cols[0].metric("Total return", f"{bt_dict['total_return']:+.2%}")
    cols[1].metric("Sharpe (after costs)", f"{bt_dict['sharpe']:.2f}")
    cols[2].metric("Max drawdown", f"{bt_dict['max_drawdown']:.2%}")
    cols[3].metric("Avg daily turnover",
                    f"${bt_dict['turnover'].mean():,.0f}")

    fig = render_equity_curve(bt_dict["equity"])
    st.plotly_chart(fig, width="stretch", height=620, key="eq_chart")
    st.caption(
        "Top: NAV over time. Bottom: drawdown from running peak (%). "
        "Dollar-neutral long-short portfolio with $1M starting NAV, "
        "5% per-name cap, integer shares, 5 bps half-spread, "
        "$0.005/share commissions."
    )
    _snapshot_button(fig, key="snap_eq",
                      default_name="equity_curve.png")

    csv_buf = StringIO()
    pd.DataFrame({
        "equity":   bt_dict["equity"],
        "pnl":      bt_dict["pnl"],
        "turnover": bt_dict["turnover"],
        "costs":    bt_dict["costs"],
    }).to_csv(csv_buf)
    st.download_button(
        "⬇  Download backtest series (CSV)",
        data=csv_buf.getvalue(),
        file_name="dynamic_factors_backtest.csv",
        mime="text/csv",
    )


# --- Methodology ---
with tab_about:
    st.markdown(
        """
        ### How the model works

        **1. PCA on daily returns.** We compute the Ledoit-Wolf shrunk
        covariance matrix of the S&P 500 daily-returns panel and take
        its top-k eigenvectors as the principal-component factors.
        Factor 1 is almost always 'the market' — uniform-sign loadings
        across the universe. Factors 2-3 typically encode sector
        rotations.

        **2. Kalman filter for time-varying loadings.** Each stock's
        loadings on the factors are NOT static. We run a Kalman filter
        per stock with state = `[β_1, β_2, …, β_k]` (the loadings),
        observation = the stock's daily return, regressors = the
        factor returns. The filter tracks the loadings as they drift
        through time — a stock's market beta during the 2020 Covid
        crash is genuinely different from its beta during a calm 2024.

        **3. Residuals.** Each filter step produces an *innovation* —
        the part of the stock's return that the (current) factor
        loadings can't explain. That innovation is the strategy's
        signal: it's the idiosyncratic component that should
        mean-revert.

        **4. Mean-reversion trade.** Compute a rolling z-score of the
        residual. When z exceeds +2 (residual is unusually high),
        short the stock; when z drops below -2, go long. Exit when
        |z| returns below 0.5. Long-short, dollar-neutral, daily
        rebalance.

        **5. Realistic costs.** Half-spread of 5 bps + $0.005/share
        commission. Integer shares, 5% NAV cap per name, $1M starting
        NAV. These eat 1-2 Sharpe points off the academic figure.

        ### What this is *not*

        * Not a real-time HFT system — yfinance quotes are delayed.
        * Not survivorship-bias-aware — the universe is today's S&P
          500 constituents applied to historical data. Don't read the
          backtest Sharpe as tradable.
        * Not a money printer. Modern statistical arbitrage Sharpe
          ratios are typically 0.5-1.5 *before* costs; after, much
          less. The story isn't "this prints money," it's "you can
          see the structure the model extracts and the residuals
          really do mean-revert."

        ### Tech stack

        * **C++17 kernel** (Eigen): covariance, eigendecomposition,
          Kalman filter (templated for fixed dim), rolling stats.
        * **pybind11**: scalar + batch APIs to numpy.
        * **Python**: yfinance + pandas + scipy for orchestration.
        * **Plotly + Streamlit**: dashboard.

        Source: [github.com/MiniKetch/dynamic-factors](https://github.com/MiniKetch/dynamic-factors)
        """
    )


st.divider()
st.caption(
    "C++ math kernel via pybind11 · Eigen for linear algebra · "
    "Data: yfinance (delayed quotes)."
)
