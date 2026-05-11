from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans

from .embed import EMBEDDING_DIM


def auto_k(n_tokens: int) -> int:
    if n_tokens < 2:
        return 1
    return max(2, min(8, math.ceil(math.sqrt(n_tokens))))


def cluster(
    embeddings: np.ndarray,
    k: int | None,
    seed: int,
) -> tuple[list[int], np.ndarray]:
    n = embeddings.shape[0]
    if n == 0:
        empty_dim = embeddings.shape[1] if embeddings.ndim > 1 else EMBEDDING_DIM
        return [], np.zeros((0, empty_dim), dtype=np.float32)

    k_effective = k if k is not None else auto_k(n)
    k_effective = min(k_effective, n)

    distinct = int(np.unique(embeddings, axis=0).shape[0])
    k_effective = min(k_effective, distinct)

    if k_effective <= 1:
        centroid = embeddings.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
        return [0] * n, centroid

    model = KMeans(n_clusters=k_effective, random_state=seed, n_init=10)
    labels = model.fit_predict(embeddings).tolist()
    return labels, model.cluster_centers_.astype(np.float32, copy=False)


_OCTANT_NAMES: dict[tuple[bool, bool, bool], str] = {
    (True,  True,  True):  "triumphant",
    (True,  True,  False): "anxious-bright",
    (True,  False, True):  "serene",
    (True,  False, False): "wistful",
    (False, True,  True):  "fierce",
    (False, True,  False): "agitated",
    (False, False, True):  "stern",
    (False, False, False): "melancholy",
}


def octant_name(centroid_vad: np.ndarray) -> str:
    v = float(centroid_vad[0])
    a = float(centroid_vad[1])
    d = float(centroid_vad[2])
    return _OCTANT_NAMES[(v >= 0, a >= 0, d >= 0)]


def cluster_with_basis(
    vectors: np.ndarray,
    k: int | None,
    seed: int,
    basis: str,
) -> tuple[list[int], np.ndarray, list[str]]:
    labels, centroids = cluster(vectors, k=k, seed=seed)
    if basis == "vad" and centroids.size > 0:
        names = [octant_name(c) for c in centroids]
    else:
        names = [f"cluster-{i}" for i in range(centroids.shape[0])]
    return labels, centroids, names
