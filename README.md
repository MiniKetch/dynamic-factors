# dynamic-factors

> Dynamic factor model for equity statistical arbitrage. PCA on the S&P 500 + Kalman-filtered time-varying loadings + realistic backtest. C++ math kernel (Eigen), pybind11 bindings, Streamlit dashboard. Runs locally — clone, install, open in your browser.

PCA on the S&P 500 daily-returns matrix extracts the dominant common factors. A Kalman filter tracks each stock's loadings on those factors as they drift through time. Trade the residuals — what each stock does *that the time-varying factor exposures can't explain*. The whole pipeline is precomputed historically and refreshed live for the current trading day.

This is a real research-grade strategy class. Two-Sigma, AQR, and most multi-strat hedge funds run versions of "stat arb with time-varying factor betas." The implementation here is the textbook version, written end-to-end with a focus on numerical correctness over throughput.

---

## What's in here

| Area | Detail |
|---|---|
| **C++17 kernel** | Linear algebra (Eigen), covariance + Ledoit-Wolf shrinkage, symmetric eigendecomposition, LDLT factorization, templated Kalman filter (k=3, k=5) with batched `run_batch` API, online rolling stats. ~100 unit assertions across 24 test cases. |
| **pybind11 bindings** | Numpy-friendly. Single FFI call per stock for the Kalman pass (`run_batch` handles the full history including per-stock NaN gaps in C++). |
| **Python data layer** | Bundled S&P 500 snapshot, batched yfinance fetcher, forward-fill alignment, log-return computation, Parquet persistence. |
| **PCA layer** | Static + rolling decomposition, sign-alignment between consecutive windows, factor identification (market / sector tilts). |
| **Kalman layer** | Per-stock loading tracker that tolerates sporadic missing trading days (no synthetic-zero fill), process-noise calibration via maximum likelihood. |
| **Backtest engine** | Z-score signals with hysteresis, dollar-neutral long-short portfolio, integer-share constraints, 5 % per-name cap, bid-ask half-spread + commission costs. |
| **Plotly viz** | Eigenvalue spectrum, factor-loading heatmap, time-evolution lines, residual / z-score panel, equity curve with drawdown. |
| **Streamlit dashboard** | 6 tabs, live data refresh, PNG snapshot per chart. |

---

## Quick start

You need a C++17 compiler, CMake 3.20+, and Python 3.10+.

**macOS:**
```bash
xcode-select --install                 # one-time, gets compiler
brew install cmake python              # if you don't have them
```

**Linux:**
```bash
sudo apt install build-essential cmake python3-pip python3-venv
```

**Windows:** Install [Visual Studio 2019+ Build Tools](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload. Bundles CMake + the compiler.

Then:

```bash
git clone https://github.com/MiniKetch/dynamic-factors.git
cd dynamic-factors

pip install -e ".[data,viz,dashboard]"        # builds the C++ kernel + pulls deps

# 1) Pull ~5 years of S&P 500 history (~5-15 minutes the first time):
python examples/run_pipeline.py

# 2) Open the dashboard:
streamlit run streamlit_app.py
```

The dashboard opens at `http://localhost:8501`. Use the sidebar to choose factor count, Kalman process noise, and signal thresholds — every chart re-renders live.

---

## A tour of the dashboard

**📊 Eigenvalue spectrum.** The hero chart. Bars show each factor's share of total variance; the gold cumulative-variance line on the secondary axis shows how many factors you need to capture most of the cross-section. The "eigenvalue cliff" — factor 1 dominates, the rest are smaller — is the canonical equity-PCA signature.

**🌡️ Factor loadings.** Heatmap of stocks × factors. Sort stocks by loading on any factor to see the cluster structure: the market factor has uniform-sign loadings across the universe, sector tilts cluster within an industry. Auto-detected labels point out which factor is the market.

**🌀 Loading evolution.** Pick any stock; see how its Kalman-filtered loadings on each factor evolved through history. Clean evidence that betas aren't static — they shift around regime breaks (Covid, rate cycles, AI rally).

**🎯 Residual & z-score.** Two-panel view of one stock's idiosyncratic returns (top) and the rolling z-score the strategy actually trades on (bottom). Threshold lines show the entry / exit decision zones.

**💰 Backtest equity.** Long-short portfolio NAV over time, plus drawdown shaded below. Summary metrics: total return, Sharpe (after costs), max drawdown, daily turnover. Download the daily series as CSV for offline analysis.

**📋 Methodology.** Plain-language explanation of the model, what it isn't, and the realistic-backtest assumptions.

---

## How the model works

**1. PCA on daily returns.** We compute the Ledoit-Wolf shrunk covariance matrix of the S&P 500 daily-returns panel and take its top-`k` eigenvectors as the principal-component factors. Shrinkage matters: with ~500 stocks and 252 trading days per year, the raw sample covariance has noisy eigenvalues. Ledoit-Wolf provides a closed-form mix between the sample covariance and a structured target.

**2. Kalman filter for time-varying loadings.** Each stock's loadings on the factors are NOT constant. We run a Kalman filter per stock with:

- state = `[β_1, β_2, …, β_k]` (the loadings)
- observation = the stock's daily return
- regressors = the factor returns

The filter tracks how the loadings drift over time. A stock's market beta during the 2020 Covid crash is genuinely different from its beta during a calm 2024.

**3. Residuals = signals.** Each filter step produces an *innovation*: the part of the stock's return that the (current) factor loadings can't explain. That innovation is the strategy's signal.

**4. Z-score mean reversion.** Compute a rolling z-score of the residual against its own past 60 days. When `z >= +2`, the residual is "too high" — short the stock. When `z <= -2`, go long. Exit when `|z| < 0.5`. Hysteresis prevents whipsawing.

**5. Realistic frictions.** Half-spread of 5 bps + $0.005/share commission. Integer shares only, 5 % NAV cap per name, $1M starting NAV. Daily rebalance. These eat 1-2 Sharpe points off the academic figure.

---

## What this is *not*

- **Not a real-time HFT system.** yfinance quotes are 15-min-delayed.
- **Not survivorship-bias-aware.** The bundled universe is today's S&P 500 constituents applied to historical data, so the backtest implicitly skips delistings. Real fix is a point-in-time constituent feed; we documented the limitation rather than papered over it.
- **Not a money printer.** Modern statistical arbitrage backtests typically Sharpe at 0.5-1.5 *before* costs; after, much less. The story is "you can see the structure the model extracts and the residuals really do mean-revert," not "this is a tradable signal."

---

## Run the C++ tests directly

If you want to exercise the kernel without going through Python:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Should print `100% tests passed` with ~24 test cases and ~100 assertions exercised.

Run the Python tests:

```bash
pip install -e ".[dev]"
pytest python/tests
```

28 tests covering PCA recovery, Kalman convergence (including process-noise calibration), signal generation, dollar-neutrality, cost models, and end-to-end backtest accounting.

---

## Architecture

```
dynamic-factors/
├── cpp/
│   ├── include/df/          public C++ headers
│   ├── src/                 linalg + PCA implementations
│   ├── tests/               doctest unit tests
│   └── bindings/            pybind11 → Python
├── python/dynamic_factors/
│   ├── data/                yfinance, alignment, S&P 500 snapshot
│   ├── factors/             PCA wrapper, identification, rolling fits
│   ├── kalman/              per-stock tracker, process-noise tuning
│   ├── backtest/            signals, portfolio, costs, engine
│   └── viz/                 dark-quant Plotly renderers
├── examples/
│   └── run_pipeline.py      offline pull → align → save Parquet
├── streamlit_app.py         the 6-tab dashboard
├── pyproject.toml           scikit-build-core
└── CMakeLists.txt           top-level CMake
```

The C++ side is pure math — no I/O, no dependencies beyond C++17 + Eigen + pybind11 + doctest. The Python side handles everything network-, file-, and visualization-related.

---

## License

MIT — see [LICENSE](LICENSE).
