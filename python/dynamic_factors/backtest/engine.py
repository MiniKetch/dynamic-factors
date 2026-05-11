"""Backtest simulation loop.

Daily rebalance:
  1. Read current signals + prices.
  2. Compute target shares (sized + dollar-neutralised).
  3. Compare to existing positions; trade the difference at
     today's price, paying transaction costs.
  4. Mark-to-market overnight at next-day prices to compute P&L.

The state machine keeps shares (signed integers per ticker), cash
(float), and a running NAV. Equity is reconstructed daily as
cash + Σ shares · price.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dynamic_factors.backtest.costs import CostModel, spread_cost_model
from dynamic_factors.backtest.portfolio import (
    enforce_dollar_neutrality,
    target_positions,
)
from dynamic_factors.backtest.signals import SignalParams, generate_signals


@dataclass
class BacktestConfig:
    initial_nav: float = 1_000_000.0
    max_per_name: float = 0.05
    rebalance_every_n_days: int = 1
    cost_model: CostModel = field(default_factory=spread_cost_model)
    signal_params: SignalParams = field(default_factory=SignalParams)
    enforce_neutrality: bool = True


@dataclass
class BacktestResult:
    """Daily series + summary stats."""
    equity: pd.Series           # NAV by date
    pnl: pd.Series              # daily $ P&L (after costs)
    turnover: pd.Series         # daily $ traded
    positions: pd.DataFrame     # shares per ticker per day
    costs: pd.Series            # daily $ cost
    config: BacktestConfig

    @property
    def total_return(self) -> float:
        """Final NAV / initial NAV − 1."""
        return float(self.equity.iloc[-1] / self.config.initial_nav - 1.0)

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe of daily simple returns. RV-relative
        to risk-free is ignored (project assumption: r=0). For a real
        Sharpe, subtract the daily risk-free rate from `daily`."""
        daily = self.pnl / self.config.initial_nav
        sd = daily.std(ddof=1)
        if not (sd > 0):
            return 0.0
        return float(daily.mean() / sd * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        """Max peak-to-trough drawdown as a fraction of NAV."""
        running_max = self.equity.cummax()
        dd = self.equity / running_max - 1.0
        return float(dd.min())


def run_backtest(
    residuals: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run the full daily-rebalance simulation.

    Parameters
    ----------
    residuals
        (T × N) DataFrame from the Kalman tracker — Kalman
        innovations per stock per day.
    prices
        (T × N) DataFrame of close prices, same index/columns.
        Used for position sizing and cost computation.
    config
        Optional config. Defaults to the BacktestConfig defaults.

    The two frames must align (same dates, same tickers).
    """
    cfg = config or BacktestConfig()

    common_dates = residuals.index.intersection(prices.index)
    common_cols  = residuals.columns.intersection(prices.columns)
    residuals = residuals.loc[common_dates, common_cols]
    prices    = prices   .loc[common_dates, common_cols]

    # Generate signals once for the whole horizon.
    signals = generate_signals(residuals, cfg.signal_params)

    # State: shares (signed), cash. Equity = cash + Σ shares · price.
    shares = pd.Series(0.0, index=common_cols)
    cash = float(cfg.initial_nav)

    equity_log: list[float]   = []
    pnl_log:    list[float]   = []
    turnover_log: list[float] = []
    cost_log:   list[float]   = []
    positions_log: list[pd.Series] = []

    prev_equity = cash

    for i, date in enumerate(common_dates):
        px_today = prices.loc[date]
        # Fill any NaN price with the previous day's value (shouldn't
        # happen on aligned data, but defensive).
        px_today = px_today.ffill()

        rebalance_today = (i % cfg.rebalance_every_n_days == 0)

        if rebalance_today:
            # Recompute target shares from today's signal.
            target = target_positions(
                signals.loc[date],
                px_today,
                nav=prev_equity,
                max_per_name=cfg.max_per_name,
            )
            if cfg.enforce_neutrality:
                target = enforce_dollar_neutrality(target, px_today)

            # Trade the difference.
            delta = target.subtract(shares, fill_value=0.0)
            trade_dollars = (delta * px_today).fillna(0.0)
            cash -= trade_dollars.sum()  # cash decreases when buying
            cost = cfg.cost_model.cost_for_trade(delta, px_today)
            cash -= cost
            shares = target.copy()
            turnover = trade_dollars.abs().sum()
        else:
            # No rebalance — drift with prices.
            cost = 0.0
            turnover = 0.0

        # Mark-to-market.
        portfolio_value = (shares * px_today).fillna(0.0).sum()
        equity = cash + portfolio_value
        pnl = equity - prev_equity

        equity_log.append(equity)
        pnl_log.append(pnl)
        turnover_log.append(turnover)
        cost_log.append(cost)
        positions_log.append(shares.copy())

        prev_equity = equity

    return BacktestResult(
        equity=pd.Series(equity_log,  index=common_dates, name="equity"),
        pnl=pd.Series(pnl_log,        index=common_dates, name="pnl"),
        turnover=pd.Series(turnover_log, index=common_dates, name="turnover"),
        costs=pd.Series(cost_log,     index=common_dates, name="costs"),
        positions=pd.DataFrame(positions_log, index=common_dates),
        config=cfg,
    )
