
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

from .config import Config
from .decision.compose import compose
from .score.render import LilyPondNotFoundError, render_lilypond
from .score.safety import assert_safe
from .score.string_backend import StringTemplateBackend
from .semantic.cluster import cluster_with_basis
from .semantic.embed import embed
from .semantic.features import TokenFeatures
from .semantic.lang import detect_language
from .semantic.phonetic import is_phonetic
from .semantic.recurrence import recurrence_matrix
from .semantic.tokenize import tokenize
from .semantic.vad import (
    MissingLexiconError,
    fill_cluster_mean_vad,
    lookup_vad_raw,
    vad_coverage,
)
from .trace.log import write_trace


def run(text_path: Path, config: Config) -> dict[str, Path]:
    text_path = Path(text_path)
    text = text_path.read_text(encoding="utf-8")
    title = text_path.stem

    tokens = tokenize(text)
    if not tokens:
        raise ValueError(f"Input {text_path} yielded no tokens — empty or whitespace-only.")

    lang = detect_language(text)
    embeddings = embed(tokens, config.embedding_model)
    R = recurrence_matrix(embeddings)

    phonetic_flags = [is_phonetic(tok) for tok in tokens]

    try:
        vad_raw, has_entry = lookup_vad_raw(
            tokens, lang, phonetic_flags=phonetic_flags
        )
        coverage = vad_coverage(has_entry)
    except MissingLexiconError as err:
        warnings.warn(
            f"{err}\nFalling back to zero VAD — dynamics will be flat and clustering "
            f"will use embeddings. Run scripts/download_lexicons.py to enable VAD.",
            RuntimeWarning,
            stacklevel=2,
        )
        vad_raw = np.full((len(tokens), 3), np.nan, dtype=np.float32)
        has_entry = np.zeros(len(tokens), dtype=bool)
        coverage = 0.0

    if coverage >= config.vad_cluster_threshold:
        cluster_basis = "vad"
        vad_for_cluster = np.nan_to_num(vad_raw, nan=0.0).astype(np.float32, copy=False)
        labels, centroids, cluster_names = cluster_with_basis(
            vad_for_cluster, k=config.k_clusters, seed=config.seed, basis="vad",
        )
    else:
        cluster_basis = "embedding"
        labels, centroids, cluster_names = cluster_with_basis(
            embeddings, k=config.k_clusters, seed=config.seed, basis="embedding",
        )

    vad = fill_cluster_mean_vad(vad_raw, has_entry, labels)

    features: list[TokenFeatures] = []
    for i, (tok, label) in enumerate(zip(tokens, labels, strict=True)):
        sim: float | None = None
        if i > 0:
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
        features.append(
            TokenFeatures(
                index=tok.index,
                text=tok.text,
                lang=lang,
                cluster_id=label,
                v=float(vad[i, 0]),
                a=float(vad[i, 1]),
                d=float(vad[i, 2]),
                sim_to_prev=sim,
                is_all_caps=tok.is_all_caps,
                rest_after=tok.rest_after,
                phonetic_mode=phonetic_flags[i],
                cluster_basis=cluster_basis,
                cluster_name=cluster_names[label] if cluster_names else f"cluster-{label}",
            )
        )

    events = compose(features, embeddings, R, centroids, vad, config)
    assert_safe(events, tempo_bpm=config.tempo_bpm)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ly_source = StringTemplateBackend().render(
        events,
        title=title,
        tempo_bpm=config.tempo_bpm,
        config_fingerprint=config.fingerprint(),
        seed=config.seed,
        model_name=config.embedding_model,
    )
    ly_path = out_dir / "score.ly"
    ly_path.write_text(ly_source, encoding="utf-8")

    pdf_path: Path | None = None
    midi_path: Path | None = None
    try:
        pdf_path, midi_path = render_lilypond(ly_path, out_dir=out_dir)
    except LilyPondNotFoundError as err:
        print(f"[semtune] warning: {err}", file=sys.stderr)
        print(
            "[semtune] .ly written successfully; install LilyPond to get .pdf and .midi.",
            file=sys.stderr,
        )

    trace_path = out_dir / "trace.md"
    write_trace(
        trace_path,
        features,
        events,
        seed=config.seed,
        model_name=config.embedding_model,
        config_fingerprint=config.fingerprint(),
        title=title,
        cluster_basis=cluster_basis,
    )

    manifest_path = out_dir / "config.json"
    manifest_path.write_text(config.to_json(), encoding="utf-8")

    result: dict[str, Path] = {
        "ly": ly_path,
        "trace": trace_path,
        "manifest": manifest_path,
    }
    if pdf_path is not None:
        result["pdf"] = pdf_path
    if midi_path is not None:
        result["midi"] = midi_path
    return result
