"""
Gate 2: retrieval + evidence-sufficiency check.

Target design is hybrid: BM25 (lexical) + dense embedding (semantic),
fused with RRF, with a disjunctive sufficiency gate (lexical_ok OR
dense_ok). Only the BM25 half is implemented so far — the dense
retriever, RRF fusion, and the disjunctive gate are still to be wired
(see EVALUATION_PLAN.md and CLAUDE.md for current status).

Rationale for hybrid on this corpus: BM25 is irreplaceable for exact
tokens the handbooks are full of (form codes like PNC:AA-FO-45,
acronyms, figures like "20%" / "5.00"), while dense retrieval covers the
register gap between how students ask and how policy is written
("can I take a break" -> "Leave of Absence"). Neither alone covers both.

This module also owns the "is the retrieved evidence good enough to
answer from" decision. It is kept separate from scope_gate.py on
purpose: retrieval weakness (corpus gap, wording mismatch) must never be
read as "out of scope".
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
        score_threshold: float = config.BM25_COVERAGE_THRESHOLD,
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