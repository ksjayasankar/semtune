from __future__ import annotations

from functools import lru_cache

import numpy as np

from .tokenize import Token

EMBEDDING_DIM = 384


@lru_cache(maxsize=4)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(name)


def embed(tokens: list[Token], model_name: str) -> np.ndarray:
    if not tokens:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = _load_model(model_name)
    texts = [t.text.lower() for t in tokens]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32, copy=False)
