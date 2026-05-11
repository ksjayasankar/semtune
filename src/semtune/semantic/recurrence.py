
from __future__ import annotations

import numpy as np


def recurrence_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    R = embeddings @ embeddings.T
    np.clip(R, -1.0, 1.0, out=R)
    return R.astype(np.float32, copy=False)
