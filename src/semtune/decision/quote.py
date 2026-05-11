
from __future__ import annotations

import numpy as np

from .events import NoteEvent
from .state import (
    NUM_DURATION_CLASSES,
    NUM_PITCH_CLASSES,
    NUM_STATES,
    state_index,
    unpack_state,
)


def find_qualifying_recurrence(
    i: int,
    R: np.ndarray,
    theta_quote: float,
    history: list[NoteEvent],
) -> tuple[int, int] | None:
    if i <= 0 or not history or i > R.shape[0]:
        return None
    past = R[i, :i]
    if past.size == 0:
        return None
    qualifying = [
        j for j in range(past.shape[0])
        if j < len(history)
        and past[j] > theta_quote
        and not history[j].is_rest
    ]
    if not qualifying:
        return None
    return qualifying[-1], qualifying[0]


def compute_target_state_cyclic(
    source_state: int,
    prev_pc: int,
    step: int,
) -> int:
    pc_src, dur_src = unpack_state(source_state)
    cycled = step % 5
    if cycled == 0 or cycled == 4:
        pc_target = pc_src
    elif cycled == 1:
        pc_target = (pc_src + 5) % NUM_PITCH_CLASSES
    elif cycled == 2:
        pc_target = (2 * prev_pc - pc_src) % NUM_PITCH_CLASSES
    elif cycled == 3:
        pc_target = (pc_src - 7) % NUM_PITCH_CLASSES
    else:
        pc_target = pc_src
    return state_index(pc_target, dur_src)


def compute_target_state_vad(
    source_state: int,
    delta_vad: np.ndarray,
) -> int:
    pc_src, dur_src = unpack_state(source_state)
    dv = float(delta_vad[0])
    da = float(delta_vad[1])
    pc_target = int(pc_src + round(dv * 6.0)) % NUM_PITCH_CLASSES
    dur_shift = int(round(da * 3.0))
    dur_target = max(0, min(NUM_DURATION_CLASSES - 1, dur_src - dur_shift))
    return state_index(pc_target, dur_target)


def eta(R: float, theta_quote: float, kappa: float) -> float:
    return float(np.exp(kappa * (R - theta_quote)))


def boost_row(row: np.ndarray, target_state: int, eta_value: float) -> np.ndarray:
    out = row.astype(np.float32, copy=True)
    out[target_state] *= float(eta_value)
    total = float(out.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(NUM_STATES, 1.0 / NUM_STATES, dtype=np.float32)
    return (out / total).astype(np.float32, copy=False)
