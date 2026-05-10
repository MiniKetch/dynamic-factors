"""Backtest engine — signals, costs, portfolio simulation."""

from dynamic_factors.backtest.signals import (
    SignalParams,
    generate_signals,
)
from dynamic_factors.backtest.portfolio import (
    target_positions,
    enforce_dollar_neutrality,
)
from dynamic_factors.backtest.costs import (
    CostModel,
    spread_cost_model,
)
from dynamic_factors.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)

__all__ = [
    "SignalParams",
    "generate_signals",
    "target_positions",
    "enforce_dollar_neutrality",
    "CostModel",
    "spread_cost_model",
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
]
