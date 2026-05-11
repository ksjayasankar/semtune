
from __future__ import annotations

NUM_PITCH_CLASSES = 12
DURATION_CLASSES: tuple[str, ...] = ("16", "8", "4", "4.", "2", "1")
DURATION_QUARTERS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0)
NUM_DURATION_CLASSES = len(DURATION_CLASSES)
NUM_STATES = NUM_PITCH_CLASSES * NUM_DURATION_CLASSES


def state_index(pc: int, dur_idx: int) -> int:
    if not 0 <= pc < NUM_PITCH_CLASSES:
        raise ValueError(f"pitch_class {pc} out of range 0..{NUM_PITCH_CLASSES - 1}")
    if not 0 <= dur_idx < NUM_DURATION_CLASSES:
        raise ValueError(f"dur_idx {dur_idx} out of range 0..{NUM_DURATION_CLASSES - 1}")
    return pc * NUM_DURATION_CLASSES + dur_idx


def unpack_state(idx: int) -> tuple[int, int]:
    if not 0 <= idx < NUM_STATES:
        raise ValueError(f"state idx {idx} out of range 0..{NUM_STATES - 1}")
    return divmod(idx, NUM_DURATION_CLASSES)


def pc_distance(a: int, b: int) -> int:
    d = abs(a - b) % NUM_PITCH_CLASSES
    return min(d, NUM_PITCH_CLASSES - d)


def octave_greedy(
    pc: int,
    prev_midi: int | None,
    soft_clamp: tuple[int, int],
    hard_clamp: tuple[int, int],
) -> int:
    anchor = prev_midi if prev_midi is not None else 60
    soft_lo, soft_hi = soft_clamp
    hard_lo, hard_hi = hard_clamp
    candidates_soft = [m for m in range(soft_lo, soft_hi + 1) if m % NUM_PITCH_CLASSES == pc]
    if candidates_soft:
        pool = candidates_soft
    else:
        pool = [m for m in range(hard_lo, hard_hi + 1) if m % NUM_PITCH_CLASSES == pc]
    best = min(pool, key=lambda m: abs(m - anchor))
    return best
