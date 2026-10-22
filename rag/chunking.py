from __future__ import annotations

import re
from dataclasses import asdict, dataclass

MAX_CHARS = 1000
OVERLAP_CHARS = 150
MIN_SECTION_CHARS = 300

_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+(\S.*?)\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:\d{1,2}\.|[•\-–*])\s+\S")
_ALLCAPS_TITLE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*\.?\s+[^a-z]{6,}$")
_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_INLINE_HEADER = re.compile(r"\b\d{1,2}\.\d{1,2}\.\s+[A-Z][A-Z0-9 ,:&+\-–]{4,}(?=\s|$)")
_SENTENCE_END = re.compile(r"(?<=[.!?:])\s+(?=[A-Z“\"(])")


@dataclass
class Chunk:
    id: int
    title: str
    chapter: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _is_heading(line: str):
    m = _HEADING.match(line)
    if not m:
        return None
    number, title = m.groups()
    if ". ." in line or len(title) > 90 or title.endswith("."):
        return None
    if " " not in title and len(title) > 25:
        return None
    if not title[0].isupper():
        return None
    if title == title.upper() and any(c.isalpha() for c in title):
        return None
    if int(number.split(".")[0]) == 0:
        return None
    return m


def clean_and_split_sections(raw: str) -> list[tuple[str, int, str]]:
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = next((i for i, l in enumerate(lines) if _is_heading(l)), 0)
    lines = lines[start:]

    sections: list[tuple[str, int, list[str]]] = []
    state = {"title": "Introduction", "chapter": 1}
    buf: list[str] = []

    def flush():
        if buf:
            sections.append((state["title"], state["chapter"], buf.copy()))
            buf.clear()

    for line in lines:
        if not line.strip() or _PAGE_NUMBER.match(line) or _ALLCAPS_TITLE.match(line):
            continue
        h = _is_heading(line)
        if h:
            flush()
            state["title"] = f"{h.group(1)} {h.group(2)}"
            state["chapter"] = int(h.group(1).split(".")[0])
            continue
        buf.append(line)
    flush()

    return [(t, c, _join_lines(ls)) for t, c, ls in sections]


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        flush_left = not line.startswith(" ")
        stripped = line.strip()
        if not out:
            out = stripped
        elif flush_left and stripped[0].islower():
            out += stripped
        elif _LIST_ITEM.match(line):
            out += "\n" + stripped
        else:
            out += " " + stripped
    out = out.replace("\\n", "\n")
    out = _INLINE_HEADER.sub(" ", out)
    return re.sub(r"[ \t]+", " ", out).strip()


def _split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [
        s.strip()
        for piece in text.split("\n")
        for s in _SENTENCE_END.split(piece)
        if s.strip()
    ]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for s in sentences:
        if current and size + len(s) + 1 > max_chars:
            pieces.append(" ".join(current))
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            current, size = tail, tail_size
        current.append(s)
        size += len(s) + 1
    if current:
        pieces.append(" ".join(current))
    if len(pieces) > 1 and len(pieces[-1]) < 200 and len(pieces[-2]) + len(pieces[-1]) <= max_chars * 1.3:
        pieces[-2] = f"{pieces[-2]} {pieces[-1]}"
        pieces.pop()
    return pieces


def build_chunks(raw: str) -> list[Chunk]:
    sections = clean_and_split_sections(raw)
    merged: list[tuple[str, int, str]] = []
    carry = ""
    for i, (title, chapter, body) in enumerate(sections):
        if carry:
            body = f"{carry}\n{body}"
            carry = ""
        is_last = i == len(sections) - 1
        if len(body) < MIN_SECTION_CHARS and not is_last:
            carry = f"{title}: {body}"
            continue
        merged.append((title, chapter, body))
    if carry:
        merged.append(("Notes", 1, carry))

    chunks: list[Chunk] = []
    for title, chapter, body in merged:
        for piece in _split_text(body):
            chunks.append(Chunk(id=len(chunks), title=title, chapter=chapter, text=piece.strip()))
    return chunks
