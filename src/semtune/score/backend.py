from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..decision.events import NoteEvent


class ScoreBackend(ABC):
    @abstractmethod
    def render(
        self,
        events: "list[NoteEvent]",
        *,
        title: str,
        tempo_bpm: int,
        config_fingerprint: str,
        seed: int,
        model_name: str,
    ) -> str: ...
