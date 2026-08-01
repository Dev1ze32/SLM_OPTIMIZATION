"""
Router: orchestrates the three gates and returns the routing decision
plus a per-component latency breakdown, matching the output contract in
prompt_and_routing_architecture.md.

    Gate 1 (scope)      -> ScopeGate      embedding similarity, no LLM call
    Gate 2 (evidence)   -> BM25Retriever  lexical retrieval, no LLM call
    Gate 3 (generation) -> RAGGenerator   the only LLM call, 0 or 1 per query

Each gate's latency is tracked in its own field on purpose: it lets later
experiments (4-bit vs higher-precision reference, base vs QLoRA-adapted)
attribute latency changes to the generator alone, not to routing overhead.
generator_calls should be 0 for "out_of_scope" and "refer" decisions and
1 for "answer" decisions — this is a cheap correctness check on the
routing table itself.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import config
from generator import RAGGenerator
from retriever import BM25Retriever, Passage
from scope_gate import ScopeGate


@dataclass
class RouteResult:
    decision: str  # "answer" | "refer" | "out_of_scope"
    answer: str
    citations: List[str] = field(default_factory=list)
    retrieved_passages: List[Passage] = field(default_factory=list)

    scope_in_scope: Optional[bool] = None
    scope_similarity: Optional[float] = None
    bm25_scores: List[float] = field(default_factory=list)

    scope_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    generator_calls: int = 0


class Router:
    def __init__(self):
        self.scope_gate = ScopeGate()
        self.retriever = BM25Retriever()
        self.generator = RAGGenerator()

    def route(self, query: str) -> RouteResult:
        total_start = time.perf_counter()

        # --- Gate 1: scope ---
        scope_decision = self.scope_gate.check(query)
        if not scope_decision.in_scope:
            return RouteResult(
                decision="out_of_scope",
                answer=config.SCOPE_MESSAGE,
                scope_in_scope=False,
                scope_similarity=scope_decision.best_similarity,
                scope_latency_ms=scope_decision.latency_ms,
                total_latency_ms=(time.perf_counter() - total_start) * 1000,
                generator_calls=0,
            )

        # --- Gate 2: evidence ---
        retrieval_result = self.retriever.retrieve(query)
        if not retrieval_result.sufficient_evidence:
            return RouteResult(
                decision="refer",
                answer=config.REFERRAL_MESSAGE,
                scope_in_scope=True,
                scope_similarity=scope_decision.best_similarity,
                bm25_scores=retrieval_result.scores,
                scope_latency_ms=scope_decision.latency_ms,
                retrieval_latency_ms=retrieval_result.latency_ms,
                total_latency_ms=(time.perf_counter() - total_start) * 1000,
                generator_calls=0,
            )

        # --- Gate 3: generation ---
        generation_result = self.generator.generate(query, retrieval_result.passages)

        return RouteResult(
            decision="answer",
            answer=generation_result.answer,
            citations=generation_result.citations,
            retrieved_passages=retrieval_result.passages,
            scope_in_scope=True,
            scope_similarity=scope_decision.best_similarity,
            bm25_scores=retrieval_result.scores,
            scope_latency_ms=scope_decision.latency_ms,
            retrieval_latency_ms=retrieval_result.latency_ms,
            generation_latency_ms=generation_result.latency_ms,
            total_latency_ms=(time.perf_counter() - total_start) * 1000,
            generator_calls=1,
        )