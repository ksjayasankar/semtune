from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9'\-]*")
_SENTENCE_END = frozenset(".!?")
_CLAUSE_END = frozenset(",;:")
_PARA_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Token:
    index: int
    text: str
    is_all_caps: bool
    rest_after: str


def tokenize(raw: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    for match in _WORD_RE.finditer(raw):
        between = raw[pos:match.start()]
        if tokens and (rest := _classify_rest(between)):
            tokens[-1] = _with_rest(tokens[-1], rest)
        pos = match.end()

        word = match.group()
        tokens.append(
            Token(
                index=len(tokens),
                text=word,
                is_all_caps=_is_all_caps(word),
                rest_after="",
            )
        )

    if tokens:
        tail = raw[pos:]
        if (rest := _classify_rest(tail)):
            tokens[-1] = _with_rest(tokens[-1], rest)

    return tokens


def _classify_rest(chunk: str) -> str | None:
    if _PARA_RE.search(chunk):
        return "paragraph"
    for ch in chunk:
        if ch in _SENTENCE_END:
            return "sentence"
    for ch in chunk:
        if ch in _CLAUSE_END:
            return "clause"
    return None


def _is_all_caps(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    if len(letters) < 2:
        return False
    return all(c.isupper() for c in letters)


def _with_rest(token: Token, rest: str) -> Token:
    order = {"": 0, "clause": 1, "sentence": 2, "paragraph": 3}
    if order[rest] > order[token.rest_after]:
        return Token(token.index, token.text, token.is_all_caps, rest)
    return token
