"""Data layer — universe management, batched fetching, alignment, persistence."""

from dynamic_factors.data.universe import (
    Universe,
    sp500_subset,
    load_bundled_universe,
)
from dynamic_factors.data.chains import fetch_prices_batched
from dynamic_factors.data.alignment import (
    align_panel,
    log_returns,
    forward_fill_limited,
)
from dynamic_factors.data.pipeline import run_pipeline, PipelineResult

__all__ = [
    "Universe",
    "sp500_subset",
    "load_bundled_universe",
    "fetch_prices_batched",
    "align_panel",
    "log_returns",
    "forward_fill_limited",
    "run_pipeline",
    "PipelineResult",
]
