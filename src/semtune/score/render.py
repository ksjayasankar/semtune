from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class LilyPondError(RuntimeError):
    pass


class LilyPondNotFoundError(LilyPondError):
    pass


def ensure_lilypond_installed() -> str:
    path = shutil.which("lilypond")
    if not path:
        raise LilyPondNotFoundError(
            "LilyPond is not on PATH. Install it (macOS: `brew install lilypond`) "
            "and re-run."
        )
    return path


def render_lilypond(ly_path: Path, *, out_dir: Path) -> tuple[Path, Path]:
    lilypond = ensure_lilypond_installed()
    ly_path = Path(ly_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ly_path.stem

    cmd = [lilypond, "--pdf", "--output", str(out_dir / stem), str(ly_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise LilyPondError(
            f"lilypond exited {result.returncode}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    pdf = out_dir / f"{stem}.pdf"
    midi = out_dir / f"{stem}.midi"
    if not pdf.exists():
        raise LilyPondError(f"Expected PDF at {pdf}, not found.")
    if not midi.exists():
        alt = out_dir / f"{stem}.mid"
        if alt.exists():
            alt.rename(midi)
        else:
            raise LilyPondError(f"Expected MIDI at {midi} or {alt}, not found.")

    return pdf, midi
