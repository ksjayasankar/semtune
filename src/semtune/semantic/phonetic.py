from __future__ import annotations

from .tokenize import Token

VOWEL_TO_PC: dict[str, int] = {
    "i": 0,  "ï": 0,  "y": 0,
    "ü": 1,
    "e": 2,  "é": 2,  "è": 2,  "ê": 2,
    "ö": 3,
    "a": 4,  "á": 4,  "à": 4,  "ä": 4,
    "o": 5,  "ó": 5,  "ò": 5,  "ô": 5,
    "u": 6,  "ú": 6,  "ù": 6,  "û": 6,
}

_PLOSIVES = frozenset("ptkbdg")
_FRICATIVES = frozenset("fvszh")


def _articulation_for(text: str) -> str:
    for ch in text.lower():
        if not ch.isalpha():
            continue
        if ch in VOWEL_TO_PC:
            continue
        if ch in _PLOSIVES:
            return "-."
        if ch in _FRICATIVES:
            return "-"
        return ""
    return ""


def phonetic_pitch_class(token_text: str) -> int:
    text_lower = token_text.lower()
    counts: dict[str, int] = {}
    order: list[str] = []
    for ch in text_lower:
        if ch in VOWEL_TO_PC:
            if ch not in counts:
                order.append(ch)
            counts[ch] = counts.get(ch, 0) + 1
    if not counts:
        return 4
    best = max(order, key=lambda v: counts[v])
    return VOWEL_TO_PC[best]


def phonetic_articulation(token_text: str) -> str:
    return _articulation_for(token_text)


def is_phonetic(token: Token, *, zipf_threshold: float = 1.0) -> bool:
    from wordfreq import zipf_frequency  # noqa: PLC0415

    if len(token.text) < 2:
        return False
    text_lower = token.text.lower()
    if zipf_frequency(text_lower, "en") >= zipf_threshold:
        return False
    if zipf_frequency(text_lower, "de") >= zipf_threshold:
        return False
    return True
