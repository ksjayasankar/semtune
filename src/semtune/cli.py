
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semtune", description=__doc__)
    p.add_argument("--text", type=Path, required=True, help="Path to input text file (UTF-8).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--out", type=Path, required=True, help="Output directory for artefacts.")
    p.add_argument("--tempo", type=int, default=96, help="Tempo in BPM (default: 96).")
    p.add_argument("--theta-quote", type=float, default=0.82,
                   help="Cosine-sim threshold for the QUOTE rule (default: 0.82).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(
        seed=args.seed,
        theta_quote=args.theta_quote,
        tempo_bpm=args.tempo,
        output_dir=args.out,
    )
    from .pipeline import run  # noqa: PLC0415

    run(text_path=args.text, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
