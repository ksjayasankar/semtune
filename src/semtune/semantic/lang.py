from __future__ import annotations


def detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect  # noqa: PLC0415
        DetectorFactory.seed = 0
        code = detect(text)
    except Exception:
        return "unknown"
    if code.startswith("en"):
        return "en"
    if code.startswith("de"):
        return "de"
    return code
