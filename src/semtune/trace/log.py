
from __future__ import annotations

from pathlib import Path

from ..decision.events import NoteEvent
from ..decision.state import DURATION_CLASSES
from ..score.string_backend import _note_to_lily
from ..semantic.features import TokenFeatures

_HEADER = """# trace — {title}

- **seed**: `{seed}`
- **model**: `{model}`
- **config fingerprint**: `{fingerprint}`
- **tokens**: {n_tokens}
- **events**: {n_events} (notes + rests)
- **cluster basis**: `{cluster_basis}`

## Cluster legend

{cluster_legend}

## Per-note decisions

| i | token | cluster | VAD | sim↑ | rule | note | dur | dyn | reason |
|---|---|---|---|---|---|---|---|---|---|
"""


def _format_vad(f: TokenFeatures) -> str:
    return f"({f.v:+.2f}, {f.a:+.2f}, {f.d:+.2f})"


def _format_sim(f: TokenFeatures) -> str:
    if f.sim_to_prev is None:
        return "—"
    return f"{f.sim_to_prev:+.2f}"


def _reason_for(event: NoteEvent, f: TokenFeatures | None) -> str:
    if event.is_rest:
        return f"rest after {f.rest_after if f else 'token'}" if f else "rest"
    if event.rule == "MARKOV":
        if f and f.sim_to_prev is not None:
            if f.sim_to_prev > 0.5:
                return "Markov, high sim → small-interval bias"
            if f.sim_to_prev < -0.3:
                return "Markov, low sim → leap bias"
        return "Markov sample from cluster matrix"
    if event.rule.startswith("MARKOV+QUOTE(cyclic"):
        return "Markov sample, recurrence force toward cyclic-development target"
    if event.rule.startswith("MARKOV+QUOTE(vad"):
        return "Markov sample, recurrence force toward ΔVAD-driven target"
    if event.rule == "PHONETIC":
        return "OOV / sound-poetry → vowel-driven pitch, consonant-driven articulation"
    return event.rule


def _build_cluster_legend(
    features: list[TokenFeatures],
    legend: dict[int, list[str]] | None,
) -> str:
    if legend:
        lines = [
            f"- cluster **{cid}**: {', '.join(sorted(set(words))[:6])}"
            for cid, words in sorted(legend.items())
        ]
        return "\n".join(lines) if lines else "(no clusters)"

    auto: dict[int, list[str]] = {}
    for f in features:
        auto.setdefault(f.cluster_id, []).append(f.text)
    lines = [
        f"- cluster **{cid}**: {', '.join(sorted(set(words))[:6])}"
        for cid, words in sorted(auto.items())
    ]
    return "\n".join(lines) if lines else "(no clusters)"


def write_trace(
    path: Path,
    features: list[TokenFeatures],
    events: list[NoteEvent],
    *,
    seed: int,
    model_name: str,
    config_fingerprint: str,
    cluster_legend: dict[int, list[str]] | None = None,
    title: str = "semtune run",
    cluster_basis: str = "embedding",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = _HEADER.format(
        title=title,
        seed=seed,
        model=model_name,
        fingerprint=config_fingerprint,
        n_tokens=len(features),
        n_events=len(events),
        cluster_basis=cluster_basis,
        cluster_legend=_build_cluster_legend(features, cluster_legend),
    )

    feature_by_index: dict[int, TokenFeatures] = {f.index: f for f in features}

    rows: list[str] = []
    for ev in events:
        f = feature_by_index.get(ev.index) if ev.index is not None else None
        token = f.text if f else "(rest)"
        if f:
            cluster = f"{f.cluster_name} (#{f.cluster_id})" if f.cluster_name else f"#{f.cluster_id}"
        else:
            cluster = "—"
        vad = _format_vad(f) if f else "—"
        sim = _format_sim(f) if f else "—"
        note_token = _note_to_lily(ev) if not ev.is_rest else "— (rest)"
        duration = DURATION_CLASSES[ev.dur_idx]
        dyn = ev.dynamic if ev.dynamic else "—"
        reason = _reason_for(ev, f)
        rows.append(
            f"| {ev.index if ev.index is not None else '—'} "
            f"| `{token}` | {cluster} | {vad} | {sim} | `{ev.rule}` "
            f"| `{note_token}` | {duration} | {dyn} | {reason} |"
        )

    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
