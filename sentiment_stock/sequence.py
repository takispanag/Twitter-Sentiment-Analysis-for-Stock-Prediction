"""Chronological sliding windows shared by current and legacy LSTM paths."""
from __future__ import annotations

import numpy as np


def sequence_windows(features: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values")
    if len(values) < lookback:
        raise ValueError(f"at least {lookback} rows are required for sequence windows")
    windows = np.stack([values[end - lookback + 1:end + 1] for end in range(lookback - 1, len(values))])
    return windows, np.arange(lookback - 1, len(values), dtype=int)
