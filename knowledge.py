"""A local knowledge base so you can feed SmartAI your OWN data.

Drop .txt, .md, .csv, or .pdf files into the `data/` folder. On startup SmartAI
loads and indexes them, then retrieves the most relevant passages to answer your
questions. This is Retrieval-Augmented Generation (RAG): the model stays the
same, but it now KNOWS your material — which is how it can beat a generic
assistant on your own topics.

No heavy dependencies: retrieval uses a self-contained TF-IDF search.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - PDF support is optional
    PdfReader = None

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".csv", ".text"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf" and PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            return ""
    return ""


def _chunk(text: str, words_per_chunk: int = 220) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


class KnowledgeBase:
    """A tiny, dependency-light retriever over your personal documents."""

    def __init__(self, folder: str = "data") -> None:
        self.folder = folder
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self._tfs: list[Counter] = []
        self._idf: dict[str, float] = {}

    def load(self) -> int:
        """Read and index every supported file in the data folder."""
        folder = Path(self.folder)
        if not folder.exists():
            return 0
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name.lower() == "readme.md":
                continue
            text = _read_file(path)
            if not text.strip():
                continue
            for chunk in _chunk(text):
                self.chunks.append(chunk)
                self.sources.append(path.name)
        self._build_index()
        return len(self.chunks)

    def _build_index(self) -> None:
        self._tfs = [Counter(_tokenize(c)) for c in self.chunks]
        df: Counter = Counter()
        for tf in self._tfs:
            for term in tf:
                df[term] += 1
        n = max(len(self.chunks), 1)
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def _vector(self, tf: Counter) -> dict[str, float]:
        return {t: f * self._idf.get(t, 0.0) for t, f in tf.items()}

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-k most relevant chunks for a query."""
        if not self.chunks:
            return []
        q_vec = self._vector(Counter(_tokenize(query)))
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scored: list[tuple[float, int]] = []
        for idx, tf in enumerate(self._tfs):
            d_vec = self._vector(tf)
            dot = sum(q_vec.get(t, 0.0) * v for t, v in d_vec.items())
            d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
            score = dot / (q_norm * d_norm)
            if score > 0:
                scored.append((score, idx))
        scored.sort(reverse=True)
        return [
            {
                "score": round(score, 3),
                "source": self.sources[idx],
                "text": self.chunks[idx],
            }
            for score, idx in scored[:k]
        ]
