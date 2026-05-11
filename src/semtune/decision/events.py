
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoteEvent:

    index: int | None
    pitch_midi: int
    pitch_class: int
    dur_idx: int
    velocity: int
    dynamic: str
    articulation: str
    is_rest: bool
    rule: str
