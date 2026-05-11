
from __future__ import annotations

import numpy as np

from ..config import Config
from ..semantic.features import TokenFeatures
from ..semantic.phonetic import phonetic_articulation, phonetic_pitch_class
from .events import NoteEvent
from .matrix import build_transition_matrix
from .quote import (
    boost_row,
    compute_target_state_cyclic,
    compute_target_state_vad,
    eta,
    find_qualifying_recurrence,
)
from .state import (
    NUM_DURATION_CLASSES,
    NUM_STATES,
    octave_greedy,
    state_index,
    unpack_state,
)
from .warp import warp_row

_DYNAMIC_BY_VALENCE = [
    (-0.75, "\\pp", 32),
    (-0.35, "\\p", 48),
    (-0.10, "\\mp", 60),
    (0.10, "\\mf", 72),
    (0.35, "\\f", 88),
    (1.01, "\\f", 92),
]


def _dynamic_for(features: TokenFeatures) -> tuple[str, int]:
    if features.is_all_caps:
        return "\\ff", 104
    for threshold, mark, vel in _DYNAMIC_BY_VALENCE:
        if features.v <= threshold:
            return mark, vel
    return "\\mf", 72


def _duration_bias_for_arousal(a: float) -> np.ndarray:
    positions = np.arange(NUM_DURATION_CLASSES, dtype=np.float32)
    centred = positions - 2.0
    return np.exp(-a * centred).astype(np.float32)


def _apply_duration_bias(row: np.ndarray, arousal: float) -> np.ndarray:
    dur_pref = _duration_bias_for_arousal(arousal)
    state_durs = np.arange(NUM_STATES) % NUM_DURATION_CLASSES
    mult = dur_pref[state_durs]
    out = row * mult
    total = out.sum()
    return out / total if total > 0 else row


def _register_for_valence(midi: int, valence: float, soft: tuple[int, int]) -> int:
    lo, hi = soft
    if valence > 0.6 and midi + 12 <= hi:
        return midi + 12
    if valence < -0.6 and midi - 12 >= lo:
        return midi - 12
    return midi


def _rest_event(rest_class: str, index: int) -> NoteEvent:
    dur_idx = {"clause": 1, "sentence": 2, "paragraph": 4}.get(rest_class, 1)
    return NoteEvent(
        index=index, pitch_midi=0, pitch_class=0, dur_idx=dur_idx,
        velocity=0, dynamic="", articulation="",
        is_rest=True, rule="REST",
    )


def compose(
    features: list[TokenFeatures],
    embeddings: np.ndarray,
    R: np.ndarray,
    cluster_centroids: np.ndarray,
    vad: np.ndarray,
    config: Config,
) -> list[NoteEvent]:
    if not features:
        return []

    k = max(1, cluster_centroids.shape[0])
    matrices = [
        build_transition_matrix(
            cluster_centroids[c] if c < cluster_centroids.shape[0] else cluster_centroids[0],
            cluster_id=c,
            seed=config.seed,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
        )
        for c in range(k)
    ]

    rng = np.random.default_rng(config.seed)
    events: list[NoteEvent] = []
    family_count: dict[int, int] = {}
    prev_state: int | None = None
    prev_midi: int | None = None

    for i, f in enumerate(features):
        dynamic, velocity = _dynamic_for(f)

        if f.phonetic_mode:
            pc = phonetic_pitch_class(f.text)
            dur_pref = _duration_bias_for_arousal(f.a)
            dur_idx = int(rng.choice(NUM_DURATION_CLASSES, p=dur_pref / dur_pref.sum()))
            midi = octave_greedy(pc, prev_midi, config.pitch_clamp_soft, config.pitch_clamp_hard)
            event = NoteEvent(
                index=f.index, pitch_midi=midi, pitch_class=pc, dur_idx=dur_idx,
                velocity=velocity, dynamic=dynamic,
                articulation=phonetic_articulation(f.text),
                is_rest=False, rule="PHONETIC",
            )
            events.append(event)
            prev_state = state_index(pc, dur_idx)
            prev_midi = midi
            if f.rest_after:
                events.append(_rest_event(f.rest_after, f.index))
            continue

        matrix = matrices[min(f.cluster_id, len(matrices) - 1)]
        if prev_state is None:
            row = matrix.mean(axis=0)
            prev_pc_for_warp = 0
        else:
            row = matrix[prev_state]
            prev_pc_for_warp, _ = unpack_state(prev_state)

        sim = f.sim_to_prev if f.sim_to_prev is not None else 1.0
        row = warp_row(row, prev_pc_for_warp, sim, config.lambda_warp)
        row = _apply_duration_bias(row, f.a)

        rule_label = "MARKOV"
        velocity_shift_from_dd = 0
        recurrence = find_qualifying_recurrence(
            i=i, R=R, theta_quote=config.theta_quote, history=events,
        )
        if recurrence is not None:
            j_star, family_key = recurrence
            source_event = events[j_star]
            source_state = state_index(source_event.pitch_class, source_event.dur_idx)
            delta_vad = vad[i] - vad[j_star]
            norm_dvad = float(np.linalg.norm(delta_vad))
            if norm_dvad < config.epsilon_vad:
                step = family_count.get(family_key, 0)
                target_state = compute_target_state_cyclic(
                    source_state=source_state,
                    prev_pc=prev_pc_for_warp,
                    step=step,
                )
                family_count[family_key] = step + 1
                rule_label = f"MARKOV+QUOTE(cyclic, step={step})"
            else:
                target_state = compute_target_state_vad(source_state, delta_vad)
                rule_label = "MARKOV+QUOTE(vad)"
                velocity_shift_from_dd = int(round(float(delta_vad[2]) * 12.0))

            row = boost_row(
                row,
                target_state,
                eta(
                    R=float(R[i, j_star]),
                    theta_quote=config.theta_quote,
                    kappa=config.kappa,
                ),
            )

        row_f64 = row.astype(np.float64)
        total = row_f64.sum()
        if total > 0:
            row_f64 = row_f64 / total
        else:
            row_f64 = np.full(NUM_STATES, 1.0 / NUM_STATES, dtype=np.float64)
        sampled = int(rng.choice(NUM_STATES, p=row_f64))
        pc, dur_idx = unpack_state(sampled)
        midi = octave_greedy(pc, prev_midi, config.pitch_clamp_soft, config.pitch_clamp_hard)
        midi = _register_for_valence(midi, f.v, config.pitch_clamp_soft)

        velocity_emit = max(20, min(120, velocity + velocity_shift_from_dd))
        event = NoteEvent(
            index=f.index, pitch_midi=midi, pitch_class=pc, dur_idx=dur_idx,
            velocity=velocity_emit, dynamic=dynamic,
            articulation="->" if f.is_all_caps else "",
            is_rest=False, rule=rule_label,
        )
        events.append(event)
        prev_state = state_index(pc, dur_idx)
        prev_midi = midi

        if f.rest_after:
            events.append(_rest_event(f.rest_after, f.index))

    return events
