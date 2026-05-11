
from __future__ import annotations

import numpy as np

from .state import NUM_DURATION_CLASSES, NUM_STATES

_STATE_PCS = np.arange(NUM_STATES) // NUM_DURATION_CLASSES


def _pc_distance_from(prev_pc: int) -> np.ndarray:
    d = np.abs(_STATE_PCS - prev_pc) % 12
    return np.minimum(d, 12 - d).astype(np.float32)


def warp_row(
    row: np.ndarray,
    prev_pc: int,
    sim: float,
    lambda_warp: float,
) -> np.ndarray:
    if sim is None or np.isnan(sim):
        total = row.sum()
        return row / total if total > 0 else row.copy()

    distances = _pc_distance_from(prev_pc)
    mid = 3.0
    warp = np.exp(lambda_warp * sim * (mid - distances)).astype(np.float32)
    warped = row * warp
    total = warped.sum()
    if total <= 0 or not np.isfinite(total):
        orig_total = row.sum()
        return row / orig_total if orig_total > 0 else row.copy()
    return (warped / total).astype(np.float32)
