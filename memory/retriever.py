"""Zero-dependency TF-IDF retriever over the company-memory corpus.

Design choices
--------------
* Pure Python (re + math) so it runs inside OpenEnv/HF Space containers
  without pulling sentence-transformers / faiss / langchain.
* Each markdown file is split into paragraph-level chunks, indexed with
  token-frequency + inverse-document-frequency, and queried with cosine
  over sparse TF-IDF vectors.
* The retriever is the *single source of grounding truth* consumed by
  both the specialists (at rollout time) and the grader (at scoring
  time), so the reward becomes verifiable instead of fuzzy.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text or "")]


@dataclass
class MemoryHit:
    source: str
    snippet: str
    score: float

    def as_citation(self) -> str:
        """Stable string form used in ExpertReport.citations and briefs."""
        return f"memory:{self.source}"


class Retriever:
    """Lightweight TF-IDF retriever over a directory of markdown files."""

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = Path(corpus_dir)
        self._docs: List[Tuple[str, str]] = []
        self._vocab: Dict[str, int] = {}
        self._tf: List[Dict[int, float]] = []
        self._df: Dict[int, int] = {}
        self._norms: List[float] = []
        self._load()
        self._build_index()

    # -- indexing ----------------------------------------------------------

    def _load(self) -> None:
        for path in sorted(self.corpus_dir.rglob("*.md")):
            rel = str(path.relative_to(self.corpus_dir)).replace("\\", "/")
            text = path.read_text(encoding="utf-8")
            chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
            for i, chunk in enumerate(chunks):
                self._docs.append((f"{rel}#chunk{i}", chunk))

    def _build_index(self) -> None:
        for _, text in self._docs:
            tokens = _tokenize(text)
            tf_doc: Dict[int, float] = {}
            for tok in tokens:
                if tok not in self._vocab:
                    self._vocab[tok] = len(self._vocab)
                idx = self._vocab[tok]
                tf_doc[idx] = tf_doc.get(idx, 0.0) + 1.0
            self._tf.append(tf_doc)
            for idx in tf_doc:
                self._df[idx] = self._df.get(idx, 0) + 1
        self._num_docs = len(self._docs)
        # cache L2 norms of tf-idf vectors for cosine similarity.
        self._norms = []
        for tf_doc in self._tf:
            sq = 0.0
            for idx, tf in tf_doc.items():
                idf = math.log((self._num_docs + 1) / (self._df.get(idx, 1) + 1)) + 1.0
                sq += (tf * idf) ** 2
            self._norms.append(math.sqrt(sq) or 1.0)

    # -- public API --------------------------------------------------------

    def query(self, text: str, k: int = 3) -> List[MemoryHit]:
        if not text or not self._docs:
            return []
        tokens = _tokenize(text)
        if not tokens:
            return []
        q_tf: Dict[int, float] = {}
        for tok in tokens:
            if tok in self._vocab:
                idx = self._vocab[tok]
                q_tf[idx] = q_tf.get(idx, 0.0) + 1.0
        if not q_tf:
            return []
        # query norm
        q_sq = 0.0
        for idx, tf in q_tf.items():
            idf = math.log((self._num_docs + 1) / (self._df.get(idx, 1) + 1)) + 1.0
            q_sq += (tf * idf) ** 2
        q_norm = math.sqrt(q_sq) or 1.0

        scores: List[Tuple[float, int]] = []
        for doc_idx, tf_doc in enumerate(self._tf):
            dot = 0.0
            for idx, qt in q_tf.items():
                tfd = tf_doc.get(idx)
                if tfd is None:
                    continue
                idf = math.log((self._num_docs + 1) / (self._df.get(idx, 1) + 1)) + 1.0
                dot += (qt * idf) * (tfd * idf)
            if dot <= 0.0:
                continue
            cosine = dot / (q_norm * self._norms[doc_idx])
            scores.append((cosine, doc_idx))
        scores.sort(reverse=True)
        hits: List[MemoryHit] = []
        for score, doc_idx in scores[: max(k, 0)]:
            source, chunk = self._docs[doc_idx]
            snippet = chunk.replace("\n", " ").strip()
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            hits.append(MemoryHit(source=source, snippet=snippet, score=round(float(score), 4)))
        return hits

    def sources(self) -> List[str]:
        return [src for src, _ in self._docs]

    def has_source(self, source: str) -> bool:
        return any(src == source for src, _ in self._docs)

    def contains_any(self, text: str, sources: Sequence[str]) -> bool:
        """Return True if any of ``sources`` appears verbatim in ``text``.

        Used by the grader to verify citations are grounded in the corpus
        rather than hallucinated strings.
        """
        if not text:
            return False
        return any(src and src in text for src in sources)

    def count_grounded_citations(self, citations: Iterable[str]) -> int:
        known = set(self.sources())
        n = 0
        for citation in citations or []:
            if not isinstance(citation, str) or not citation.startswith("memory:"):
                continue
            if citation[len("memory:"):] in known:
                n += 1
        return n


__all__ = ["MemoryHit", "Retriever"]
