from __future__ import annotations

import json
from pathlib import Path

from .chunking import build_chunks

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "data" / "pe_notes.txt"
CHUNKS = ROOT / "data" / "chunks.json"


def build_and_save() -> list[dict]:
    chunks = [c.to_dict() for c in build_chunks(NOTES.read_text(encoding="utf-8"))]
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    return chunks


def load_chunks() -> list[dict]:
    if CHUNKS.exists():
        return json.loads(CHUNKS.read_text(encoding="utf-8"))
    return [c.to_dict() for c in build_chunks(NOTES.read_text(encoding="utf-8"))]
