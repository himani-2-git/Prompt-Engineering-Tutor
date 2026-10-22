from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = frozenset("""
a an and are as at be but by can could do does did explain for from give had has have how i if in into is it its
me my of on or please tell that the their them then there these this those to use used using was we were what when
where which who whom why will with would you your define describe mean means meaning about between difference
differences vs versus
""".split())

_SMALL_WORDS = {"of", "and", "the", "for", "to", "in", "a", "an", "vs"}


def stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"), ("xes", "x")):
        if word.endswith(suf):
            word = word[: -len(suf)] + rep
            break
    else:
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
    for suf in ("ing", "ed", "ly"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def extract_acronyms(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for m in re.finditer(r"((?:[A-Za-z][\w\-]*\s+){1,8})\(([A-Z][A-Za-z]{1,7})\)", text):
        words = m.group(1).split()
        abbr = m.group(2)
        targets = {abbr.lower(), abbr.lower().rstrip("s")}
        for n in range(1, min(len(words), len(abbr) + 2) + 1):
            tail = words[-n:]
            full = "".join(w[0] for w in tail).lower()
            core = "".join(w[0] for w in tail if w.lower() not in _SMALL_WORDS).lower()
            if full in targets or core in targets:
                phrase_tokens = tokenize(" ".join(tail))
                if phrase_tokens:
                    for t in targets:
                        found.setdefault(t, phrase_tokens)
                break
    return found


@dataclass
class Hit:
    chunk: dict
    score: float


class Retriever:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c["title"]) * 2 + tokenize(c["text"]) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        corpus = " ".join(c["text"] + " " + c["title"] for c in chunks)
        self.acronyms = extract_acronyms(corpus)
        self._phrases = {tuple(v): k for k, v in self.acronyms.items() if len(v) > 1}

    def expand(self, tokens: list[str]) -> list[str]:
        extra: list[str] = []
        for t in tokens:
            extra += self.acronyms.get(t, [])
        joined = " ".join(tokens)
        for phrase, abbr in self._phrases.items():
            if " ".join(phrase) in joined:
                extra.append(abbr)
        return list(dict.fromkeys(tokens + extra))

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q_tokens = self.expand(tokenize(query))
        if not q_tokens:
            return []
        scores = [0.0] * len(self.chunks)
        for t in q_tokens:
            idf = self.idf.get(t, 0.0)
            if idf <= 0:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if not f:
                    continue
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * (self.lengths[i] / self.avg_len))
                scores[i] += idf * (num / den)
        best = sorted(
            [Hit(self.chunks[i], s) for i, s in enumerate(scores) if s > 0],
            key=lambda h: h.score,
            reverse=True,
        )
        return best[:k]
