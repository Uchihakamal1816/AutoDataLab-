"""Organizational memory (RAG pillar) for AutoDataLab Solo.

Exposes a module-level retriever keyed at ``memory/`` so every specialist
and the grader can share the same corpus index without passing it through
every expert signature.
"""
from __future__ import annotations

from pathlib import Path

from .retriever import MemoryHit, Retriever

_CORPUS_DIR = Path(__file__).resolve().parent
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Return the process-wide retriever, building it lazily on first use."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever(_CORPUS_DIR)
    return _retriever


def corpus_dir() -> Path:
    return _CORPUS_DIR


__all__ = ["MemoryHit", "Retriever", "get_retriever", "corpus_dir"]
