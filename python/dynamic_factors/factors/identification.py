"""Factor labelling.

PCA gives you eigenvectors but no semantics. The market factor is
*usually* factor_0 (largest eigenvalue, all-positive loadings), but
"usually" isn't always — and for content / dashboard purposes we
want labels like "market", "size", "sector" instead of "factor_0".

This module attaches interpretable labels by checking each factor's
correlation with known reference series (cap-weighted basket as a
market proxy, sector dummies for sector tilts, etc.).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from dynamic_factors.factors.pca import PCAResult


def label_factors(
    pca: PCAResult,
    *,
    sector_map: Optional[pd.Series] = None,
) -> pd.Series:
    """Return a Series mapping ``factor_i`` → human-readable label.

    Heuristic order:
      1. Factor with all-positive loadings ⇒ "market".
      2. Factor whose loadings split cleanly along a sector
         dimension ⇒ "sector: <name>".
      3. Anything else ⇒ keep the raw "factor_i" label.

    ``sector_map`` is an optional Series indexed by ticker, with
    sector strings as values. If absent, only the market detection
    runs.
    """
    labels = {}
    for col in pca.loadings.columns:
        loadings = pca.loadings[col]

        # Market: all loadings same sign and roughly uniform.
        if (loadings > 0).all() or (loadings < 0).all():
            labels[col] = "market"
            continue
        # Or: dominant component's sign agrees on > 80 % of stocks.
        positive_share = float((loadings > 0).mean())
        if positive_share > 0.8 or positive_share < 0.2:
            labels[col] = "market"
            continue

        # Sector: highest |loading| dispersion within a sector that
        # is most explained by this factor.
        if sector_map is not None:
            common = loadings.index.intersection(sector_map.index)
            if len(common) > 5:
                grouped = pd.Series(loadings[common].values,
                                     index=sector_map.loc[common].values).abs()
                sector_means = grouped.groupby(level=0).mean()
                if not sector_means.empty:
                    top_sector = sector_means.idxmax()
                    if sector_means[top_sector] > 1.4 * sector_means.median():
                        labels[col] = f"sector: {top_sector}"
                        continue

        labels[col] = col  # fallback
    return pd.Series(labels, name="factor_label")
