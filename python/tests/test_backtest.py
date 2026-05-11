"""Backtest engine tests — signals + portfolio + costs + engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamic_factors.backtest import (
    BacktestConfig, CostModel, SignalParams,
    enforce_dollar_neutrality, generate_signals,
    run_backtest, target_positions,
)
from dynamic_factors.backtest.signals import rolling_zscore


@pytest.fixture
def small_residuals():
    """Two stocks with mean-reverting deviations baked in."""
    rng = np.random.default_rng(13)
    T = 200
    idx = pd.bdate_range("2024-01-01", periods=T)
    rets = pd.DataFrame(
        rng.standard_normal((T, 2)) * 0.01,
        index=idx, columns=["A", "B"],
    )
    # Inject one large positive deviation in stock A around day 100.
    rets.loc[idx[100], "A"] = 0.10
    return rets


def test_rolling_zscore_shape(small_residuals):
    z = rolling_zscore(small_residuals, window=30, min_periods=20)
    assert z.shape == small_residuals.shape


def test_generate_signals_emits_minus_one_after_positive_spike(small_residuals):
    """The day after a +10% deviation, with z far above entry,
    the signal should be -1 (short the rich stock)."""
    sigs = generate_signals(
        small_residuals,
        SignalParams(window=30, entry_threshold=2.0, exit_threshold=0.5),
    )
    # The spike day is 100; the z-score includes it, so the signal
    # for stock A on that day should be -1.
    assert sigs.loc[small_residuals.index[100], "A"] == -1.0


def _generate_signals_loop(residuals, params):
    """Reference implementation — the pre-refactor iterrows() loop.

    Kept solely as a regression oracle: the vectorised version must
    produce bitwise-identical output to this on any input.
    """
    z = rolling_zscore(residuals, window=params.window,
                       min_periods=params.min_window)
    pos = pd.DataFrame(0.0, index=residuals.index, columns=residuals.columns)
    prev = pd.Series(0.0, index=residuals.columns)
    for t, row in z.iterrows():
        new = prev.copy()
        new[row >=  params.entry_threshold] = -1.0
        new[row <= -params.entry_threshold] = +1.0
        new[row.abs() <= params.exit_threshold] = 0.0
        new[row.isna()] = 0.0
        pos.loc[t] = new
        prev = new
    return pos


def test_generate_signals_matches_loop_reference():
    """The vectorised generate_signals must equal the original loop
    output on every cell of a representative synthetic residual matrix."""
    rng = np.random.default_rng(2026)
    T, N = 250, 8
    idx = pd.bdate_range("2024-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    resid = pd.DataFrame(
        rng.standard_normal((T, N)) * 0.02,
        index=idx, columns=cols,
    )
    # Inject occasional spikes so signals actually fire.
    resid.iloc[50, 0] = 0.20
    resid.iloc[120, 3] = -0.15
    resid.iloc[180, 5] = 0.10

    params = SignalParams(window=30, entry_threshold=2.0,
                          exit_threshold=0.5, min_window=20)
    sigs_vec = generate_signals(resid, params)
    sigs_loop = _generate_signals_loop(resid, params)
    assert np.array_equal(sigs_vec.values, sigs_loop.values)
    assert list(sigs_vec.index) == list(sigs_loop.index)
    assert list(sigs_vec.columns) == list(sigs_loop.columns)


def test_target_positions_respects_per_name_cap():
    """When the equal-split per-name allocation exceeds the cap, the
    cap should bind: each leg gets max_per_name × nav, not nav / n."""
    signals = pd.Series([1.0, 1.0, 1.0, 1.0], index=["A", "B", "C", "D"])
    prices  = pd.Series([100.0, 100.0, 100.0, 100.0],
                          index=["A", "B", "C", "D"])
    # 4 active @ $1M NAV → equal split = $250k each (25 %). Cap at
    # 10 % per name = $100k → 1,000 shares — binds.
    shares = target_positions(signals, prices, nav=1_000_000.0,
                                max_per_name=0.10)
    assert (shares == 1000.0).all()


def test_target_positions_uses_equal_split_when_cap_loose():
    """If the cap is loose (e.g. 30 %), the equal split (25 %) wins."""
    signals = pd.Series([1.0, 1.0, 1.0, 1.0], index=["A", "B", "C", "D"])
    prices  = pd.Series([100.0, 100.0, 100.0, 100.0],
                          index=["A", "B", "C", "D"])
    shares = target_positions(signals, prices, nav=1_000_000.0,
                                max_per_name=0.30)
    # 4 active → $1M / 4 = $250k → 2,500 shares.
    assert (shares == 2500.0).all()


def test_dollar_neutrality_balances_legs():
    shares = pd.Series({"A": 100.0, "B": -50.0, "C": 200.0, "D": -100.0})
    prices = pd.Series({"A": 50.0, "B": 100.0, "C": 30.0, "D": 100.0})
    out = enforce_dollar_neutrality(shares, prices)
    long_dollars  = (out[out > 0] * prices[out > 0]).sum()
    short_dollars = -(out[out < 0] * prices[out < 0]).sum()
    # After balancing they should be approximately equal (within
    # rounding to integer shares).
    assert abs(long_dollars - short_dollars) < 100.0


def test_cost_model_charges_spread_and_commission():
    cm = CostModel(half_spread_bps=10.0, commission_per_share=0.01)
    # Trading 100 shares of $50 stock = $5,000 notional × 10 bps = $5
    # plus 100 × $0.01 = $1 commission = $6 total.
    cost = cm.cost_for_trade(
        pd.Series({"A": 100.0}), pd.Series({"A": 50.0}),
    )
    assert cost == pytest.approx(6.0, abs=1e-9)


def test_run_backtest_smoke(small_residuals):
    """End-to-end engine doesn't crash and produces sensible shapes."""
    prices = (1.0 + small_residuals).cumprod() * 100.0
    result = run_backtest(small_residuals, prices)
    assert result.equity.shape == (len(small_residuals),)
    assert result.equity.iloc[0] == pytest.approx(
        result.config.initial_nav, rel=1e-6,
    )
    # Check summary stats are finite.
    assert np.isfinite(result.total_return)
    assert np.isfinite(result.sharpe)
    assert np.isfinite(result.max_drawdown)
