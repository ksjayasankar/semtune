
from __future__ import annotations

import numpy as np

from .state import NUM_DURATION_CLASSES, NUM_PITCH_CLASSES, NUM_STATES

_PC_DIST = np.array(
    [
        [min(abs(a - b), NUM_PITCH_CLASSES - abs(a - b)) for b in range(NUM_PITCH_CLASSES)]
        for a in range(NUM_PITCH_CLASSES)
    ],
    dtype=np.float32,
)


def _centroid_to_pc_bias(centroid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((NUM_PITCH_CLASSES, centroid.shape[0])).astype(np.float32)
    raw = projection @ centroid.astype(np.float32)
    stabilised = raw - raw.max()
    exp = np.exp(stabilised)
    return (exp / exp.sum()).astype(np.float32)


def _state_pc_dur_arrays() -> tuple[np.ndarray, np.ndarray]:
    pcs = np.arange(NUM_STATES) // NUM_DURATION_CLASSES
    durs = np.arange(NUM_STATES) % NUM_DURATION_CLASSES
    return pcs.astype(np.int32), durs.astype(np.int32)


_STATE_PCS, _STATE_DURS = _state_pc_dur_arrays()


def build_transition_matrix(
    cluster_centroid: np.ndarray,
    cluster_id: int,
    seed: int,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    pc_bias = _centroid_to_pc_bias(cluster_centroid, seed + cluster_id * 9973)

    from_pcs = _STATE_PCS[:, None]
    to_pcs = _STATE_PCS[None, :]
    from_durs = _STATE_DURS[:, None]
    to_durs = _STATE_DURS[None, :]

    pc_cost = _PC_DIST[from_pcs, to_pcs]
    dur_cost = np.abs(from_durs - to_durs).astype(np.float32)
    bias = pc_bias[to_pcs]

    logits = -alpha * pc_cost - beta * dur_cost + gamma * bias
    logits -= logits.max(axis=1, keepdims=True)
    T = np.exp(logits).astype(np.float32)
    T /= T.sum(axis=1, keepdims=True)
    return T
