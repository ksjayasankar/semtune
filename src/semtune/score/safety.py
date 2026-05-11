from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..decision.events import NoteEvent

PITCH_HARD_LO = 21
PITCH_HARD_HI = 108
VELOCITY_LO = 20
VELOCITY_HI = 120
TEMPO_LO = 40
TEMPO_HI = 200


class PianoSafetyError(ValueError):
    pass


def assert_safe(events: "list[NoteEvent]", *, tempo_bpm: int) -> None:
    if not TEMPO_LO <= tempo_bpm <= TEMPO_HI:
        raise PianoSafetyError(f"tempo_bpm={tempo_bpm} out of [{TEMPO_LO}..{TEMPO_HI}]")

    for e in events:
        if e.is_rest:
            continue
        if not PITCH_HARD_LO <= e.pitch_midi <= PITCH_HARD_HI:
            raise PianoSafetyError(
                f"event {e.index}: pitch_midi={e.pitch_midi} outside piano range "
                f"[{PITCH_HARD_LO}..{PITCH_HARD_HI}]"
            )
        if not VELOCITY_LO <= e.velocity <= VELOCITY_HI:
            raise PianoSafetyError(
                f"event {e.index}: velocity={e.velocity} outside [{VELOCITY_LO}..{VELOCITY_HI}]"
            )
