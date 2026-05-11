from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenFeatures:
    index: int
    text: str
    lang: str
    cluster_id: int
    v: float
    a: float
    d: float
    sim_to_prev: float | None
    is_all_caps: bool
    rest_after: str
    phonetic_mode: bool
    cluster_basis: str = "embedding"
    cluster_name: str = ""
