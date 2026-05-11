
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from json import dumps
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 42

    theta_quote: float = 0.82

    alpha: float = 0.4
    beta: float = 0.2
    gamma: float = 0.6

    lambda_warp: float = 1.2

    kappa: float = 20.0
    epsilon_vad: float = 0.10
    vad_cluster_threshold: float = 0.5

    tempo_bpm: int = 96

    k_clusters: int | None = None

    pitch_clamp_soft: tuple[int, int] = (45, 84)
    pitch_clamp_hard: tuple[int, int] = (21, 108)

    order: int = 2

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    output_dir: Path = field(default_factory=lambda: Path("output"))

    def fingerprint(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "output_dir"}
        return sha256(dumps(payload, sort_keys=True).encode()).hexdigest()[:8]

    def to_json(self) -> str:
        payload = {**asdict(self), "output_dir": str(self.output_dir)}
        return dumps(payload, sort_keys=True, indent=2)
