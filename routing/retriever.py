"""
Gate 2: BM25 retrieval + evidence-sufficiency check.

BM25 is the sole retriever per the locked study decisions (no dense
retrieval, RRF, or reranking). This module also owns the "is the
retrieved evidence good enough to answer from" decision. It is kept
separate from scope_gate.py on purpose: retrieval weakness (corpus gap,
wording mismatch) must never be read as "out of scope".
"""

import json
import time
from dataclasses import dataclass, field
from typing import List

from rank_bm25 import BM25Okapi

import config


@dataclass
class Passage:
    document_id: str
    title: str
    source: str
    revision_date: str
    page_or_section: str
    text: str


@dataclass
class RetrievalResult:
    sufficient_evidence: bool
    passages: List[Passage] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(
        self,
        corpus_path: str = config.CORPUS_PATH,
        top_k: int = config.BM25_TOP_K,
        score_threshold: float = config.BM25_SCORE_THRESHOLD,
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold

        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)["passages"]

        self.passages = [Passage(**doc) for doc in raw_docs]
        tokenized_corpus = [_tokenize(p.text) for p in self.passages]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str) -> RetrievalResult:
        start = time.perf_counter()

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.passages, scores), key=lambda pair: pair[1], reverse=True
        )
        top = ranked[: self.top_k]

        top_scores = [float(s) for _, s in top]
        sufficient = bool(top_scores) and top_scores[0] >= self.score_threshold
        top_passages = [p for p, _ in top] if sufficient else []

        latency_ms = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            sufficient_evidence=sufficient,
            passages=top_passages,
            scores=top_scores,
            latency_ms=latency_ms,
        )