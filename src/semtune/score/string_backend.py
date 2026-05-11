from __future__ import annotations

from ..decision.events import NoteEvent
from ..decision.state import DURATION_CLASSES
from .backend import ScoreBackend

_PC_TO_LILY: tuple[str, ...] = (
    "c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b",
)


def _octave_marker(midi: int) -> str:
    octave_number = midi // 12 - 1
    if octave_number >= 3:
        return "'" * (octave_number - 3)
    return "," * (3 - octave_number)


def _note_to_lily(event: NoteEvent) -> str:
    duration = DURATION_CLASSES[event.dur_idx]
    if event.is_rest:
        return f"r{duration}"

    pc_name = _PC_TO_LILY[event.pitch_class]
    octave = _octave_marker(event.pitch_midi)
    token = f"{pc_name}{octave}{duration}"
    if event.articulation:
        token += event.articulation
    if event.dynamic:
        token += event.dynamic
    return token


class StringTemplateBackend(ScoreBackend):
    def render(
        self,
        events: list[NoteEvent],
        *,
        title: str,
        tempo_bpm: int,
        config_fingerprint: str,
        seed: int,
        model_name: str,
    ) -> str:
        body_tokens = [_note_to_lily(e) for e in events]

        lines: list[str] = []
        chunk: list[str] = []
        for tok in body_tokens:
            chunk.append(tok)
            if len(chunk) >= 4:
                lines.append("      " + " ".join(chunk))
                chunk = []
        if chunk:
            lines.append("      " + " ".join(chunk))
        body = "\n".join(lines)

        header_tagline = (
            f"seed={seed} config={config_fingerprint} model={model_name}"
        )

        return _TEMPLATE.format(
            title=_lily_escape(title),
            tagline=_lily_escape(header_tagline),
            tempo_bpm=tempo_bpm,
            body=body,
        )


def _lily_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', r"\"")


_TEMPLATE = r"""\version "2.24.0"

\header {{
  title = "{title}"
  composer = "semtune — semantic-driven score generation"
  tagline = "{tagline}"
}}

\score {{
  \new Staff \with {{
    midiInstrument = #"acoustic grand"
  }} {{
    \clef treble
    \key c \major
    \tempo 4 = {tempo_bpm}

{body}
  }}
  \layout {{ }}
  \midi {{ }}
}}
"""
