from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .tokenize import Token

CACHE_DIR = Path.home() / ".cache" / "semtune"


class MissingLexiconError(FileNotFoundError):
    def __init__(self, lang: str, path: Path) -> None:
        super().__init__(
            f"NRC-VAD lexicon for '{lang}' not found at {path}.\n"
            f"Run: python scripts/download_lexicons.py"
        )
        self.lang = lang
        self.path = path


@dataclass(frozen=True)
class _LangLexicon:
    lang: str
    mapping: dict[str, tuple[float, float, float]]


@lru_cache(maxsize=4)
def _load_lexicon(lang: str) -> _LangLexicon:
    path = CACHE_DIR / f"nrc_vad_{lang}.tsv"
    if not path.exists():
        raise MissingLexiconError(lang, path)

    mapping: dict[str, tuple[float, float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            word = parts[0].strip().lower()
            try:
                v = float(parts[1]) * 2 - 1
                a = float(parts[2]) * 2 - 1
                d = float(parts[3]) * 2 - 1
            except ValueError:
                continue
            mapping[word] = (v, a, d)
    return _LangLexicon(lang=lang, mapping=mapping)


def _lookup_one(word: str, lang: str) -> tuple[float, float, float] | None:
    try:
        lex = _load_lexicon(lang)
    except MissingLexiconError:
        raise
    return lex.mapping.get(word.lower())


def lookup_vad_raw(
    tokens: list[Token],
    lang: str,
    *,
    phonetic_flags: list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(tokens)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=bool)

    phonetic_flags = phonetic_flags or [False] * n
    primary = "de" if lang == "de" else "en"
    secondary = "en" if primary == "de" else "de"

    _load_lexicon(primary)
    _load_lexicon(secondary)

    vad = np.full((n, 3), np.nan, dtype=np.float32)
    has_entry = np.zeros(n, dtype=bool)
    for i, tok in enumerate(tokens):
        if phonetic_flags[i]:
            continue
        hit = _lookup_one(tok.text, primary) or _lookup_one(tok.text, secondary)
        if hit is not None:
            vad[i] = hit
            has_entry[i] = True
    return vad, has_entry


def vad_coverage(has_entry: np.ndarray) -> float:
    if has_entry.size == 0:
        return 0.0
    return float(has_entry.sum()) / float(has_entry.size)


def fill_cluster_mean_vad(
    vad_raw: np.ndarray,
    has_entry: np.ndarray,
    cluster_ids: list[int],
) -> np.ndarray:
    if vad_raw.size == 0:
        return vad_raw.copy()
    out = vad_raw.copy()
    cluster_ids_arr = np.asarray(cluster_ids)
    missing = ~has_entry
    for c in np.unique(cluster_ids_arr):
        members_known = (cluster_ids_arr == c) & has_entry
        if members_known.any():
            mean_vad = out[members_known].mean(axis=0)
        else:
            mean_vad = np.zeros(3, dtype=np.float32)
        targets = (cluster_ids_arr == c) & missing
        out[targets] = mean_vad
    np.nan_to_num(out, copy=False)
    return out.astype(np.float32, copy=False)
