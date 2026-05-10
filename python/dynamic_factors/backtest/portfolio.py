"""Position sizing and dollar-neutrality enforcement.

The signal matrix gives us +1/-1/0 per stock. The portfolio layer
turns those signals into actual share counts — sized within a NAV
budget, capped at a per-name percentage, and force-balanced so
the long book and short book are dollar-equal (true beta-neutral
needs more work; dollar-neutral is the close-enough version most
retail backtests use).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def target_positions(
    signals: pd.Series,
    prices: pd.Series,
    *,
    nav: float,
    max_per_name: float = 0.05,
) -> pd.Series:
    """One-day position sizer. Returns share counts (signed).

    Each *active* signal (±1) gets an equal share of the NAV
    budget, capped at ``max_per_name`` of NAV. Then we round to
    integer shares (long-only and short-only legs separately) and
    return.
    """
    if signals.shape != prices.shape:
        raise ValueError("signals and prices must align")
    active = signals[signals != 0]
    if active.empty:
        return pd.Series(0, index=signals.index, dtype=float)

    n_active = len(active)
    per_name_budget = nav / n_active
    per_name_cap = max_per_name * nav
    budget = min(per_name_budget, per_name_cap)

    # Dollar exposure per leg, signed.
    dollar = active * budget
    px_active = prices.loc[active.index]
    # Integer-share rounding, preserving sign.
    shares = (dollar / px_active).round().astype(float)

    full = pd.Series(0.0, index=signals.index)
    full.loc[shares.index] = shares
    return full


def enforce_dollar_neutrality(
    shares: pd.Series,
    prices: pd.Series,
) -> pd.Series:
    """Scale the smaller leg to match the larger leg in dollar terms.

    Common practice: long book ≈ short book in $. We scale whichever
    side is smaller (in absolute dollars) up to match the larger.
    Then re-round to integer shares.

    If one side is empty (e.g., all shorts that day), we leave it
    alone — the strategy isn't dollar-neutral that day, but
    forcing a fake offset would be worse.
    """
    long_mask  = shares > 0
    short_mask = shares < 0
    if not long_mask.any() or not short_mask.any():
        return shares

    long_value  = (shares[long_mask]  * prices[long_mask]).sum()
    short_value = -(shares[short_mask] * prices[short_mask]).sum()  # positive
    if long_value <= 0 or short_value <= 0:
        return shares

    out = shares.copy()
    if long_value > short_value:
        scale = short_value / long_value
        out.loc[long_mask] = (shares[long_mask] * scale).round()
    else:
        scale = long_value / short_value
        out.loc[short_mask] = (shares[short_mask] * scale).round()
    return out
