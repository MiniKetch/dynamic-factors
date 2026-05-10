"""Transaction cost models.

Real frictions on a long-short equity strategy come in two pieces:
  * **Half-spread** — when you cross the bid-ask spread to enter or
    exit, you pay roughly half the spread per share.
  * **Commissions** — typically a per-share fee.

For a stat-arb strategy that turns over a sizeable fraction of NAV
daily, the cumulative cost is the dominant driver of "before vs
after costs" Sharpe — easily 1-2 Sharpe points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """Parameters for the spread-plus-commissions model."""
    half_spread_bps: float = 5.0     # half-spread in bps of price
    commission_per_share: float = 0.005  # $ per share

    def cost_for_trade(self, shares_traded: pd.Series,
                       prices: pd.Series) -> float:
        """Total $ cost of trading `shares_traded` shares at `prices`.

        ``shares_traded`` is positive for buys, negative for sells —
        but for cost purposes only the *absolute* value matters.
        """
        if shares_traded.empty:
            return 0.0
        abs_shares = shares_traded.abs()
        notional = (abs_shares * prices).sum()
        spread = notional * (self.half_spread_bps / 10_000.0)
        commish = abs_shares.sum() * self.commission_per_share
        return float(spread + commish)


def spread_cost_model(
    half_spread_bps: float = 5.0,
    commission_per_share: float = 0.005,
) -> CostModel:
    """Convenience constructor."""
    return CostModel(
        half_spread_bps=half_spread_bps,
        commission_per_share=commission_per_share,
    )
