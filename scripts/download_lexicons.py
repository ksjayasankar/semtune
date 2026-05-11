
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "semtune"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENGLISH_URL = (
    "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"
)
GERMAN_URL = ENGLISH_URL


def download_zip(url: str) -> zipfile.ZipFile:
    sys.stderr.write(f"[semtune] downloading {url}\n")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        data = resp.read()
    return zipfile.ZipFile(io.BytesIO(data))


def extract_english_tsv(zf: zipfile.ZipFile, dest: Path) -> None:
    candidates = [
        name for name in zf.namelist()
        if name.lower().endswith(".txt") and "english" in name.lower()
        or name.lower().endswith("nrc-vad-lexicon.txt")
    ]
    if not candidates:
        candidates = [n for n in zf.namelist() if n.lower().endswith(".txt")]
    if not candidates:
        raise SystemExit("No .txt found inside the NRC-VAD zip archive.")
    chosen = candidates[0]
    sys.stderr.write(f"[semtune] extracting {chosen} → {dest}\n")
    with zf.open(chosen) as f:
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    clean = [ln for ln in lines if ln and not ln.startswith("Word")]
    dest.write_text("\n".join(clean) + "\n", encoding="utf-8")


def extract_translation(zf: zipfile.ZipFile, lang_keyword: str, dest: Path) -> None:
    candidates = [
        name for name in zf.namelist()
        if lang_keyword.lower() in name.lower() and name.lower().endswith(".txt")
    ]
    if not candidates:
        sys.stderr.write(
            f"[semtune] no translation file found for '{lang_keyword}' in archive.\n"
            f"[semtune] Expected a file whose name contains '{lang_keyword}'.\n"
            f"[semtune] Files in archive: {zf.namelist()[:20]}...\n"
        )
        raise SystemExit(1)
    chosen = candidates[0]
    sys.stderr.write(f"[semtune] extracting {chosen} → {dest}\n")
    with zf.open(chosen) as f:
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    clean = [ln for ln in lines if ln and not ln.startswith("Word")]
    dest.write_text("\n".join(clean) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["en", "de", "both"], default="both")
    args = parser.parse_args(argv)

    dest_en = CACHE_DIR / "nrc_vad_en.tsv"
    dest_de = CACHE_DIR / "nrc_vad_de.tsv"

    if args.lang in ("en", "both") and not dest_en.exists():
        zf = download_zip(ENGLISH_URL)
        extract_english_tsv(zf, dest_en)

    if args.lang in ("de", "both") and not dest_de.exists():
        zf = download_zip(GERMAN_URL)
        extract_translation(zf, "german", dest_de)

    sys.stderr.write(f"[semtune] done. Cache dir: {CACHE_DIR}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
