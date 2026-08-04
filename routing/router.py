"""
Router: orchestrates the three gates and returns the routing decision
plus a per-component latency breakdown, matching the output contract in
prompt_and_routing_architecture.md.

    Gate 2 (evidence)   -> BM25Retriever  lexical retrieval, no LLM call
    Gate 1 (scope)      -> ScopeGate      embedding similarity, no LLM call
    Gate 3 (generation) -> RAGGenerator   the only LLM call, 0 or 1 per query

Retrieval runs first. Gate 1's decision is only ACTED on when retrieval
comes up short — a query retrieval can already answer is never
second-guessed by the scope check. This closes a false-out-of-scope
failure mode: under the old scope-first order, a paraphrased in-scope
query (e.g. "can I take a break?") could be rejected before retrieval
ever got a chance to match it against "Leave of Absence".

Gate 1 still runs on every query, sufficient or not, so its similarity
score is always logged (see scope_similarity below) even on the answer
path. That log is what lets an evaluation report how often the old
ordering would have wrongly rejected an answerable query.

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

        # --- Gate 2: evidence ---
        retrieval_result = self.retriever.retrieve(query)

        # --- Gate 1: scope ---
        # Always runs (cheap, CPU-only) so scope_similarity is logged on
        # every path, but its decision is only consulted below when
        # retrieval alone isn't enough to answer.
        scope_decision = self.scope_gate.check(query)

        if retrieval_result.sufficient_evidence:
            # --- Gate 3: generation ---
            generation_result = self.generator.generate(query, retrieval_result.passages)

            return RouteResult(
                decision="answer",
                answer=generation_result.answer,
                citations=generation_result.citations,
                retrieved_passages=retrieval_result.passages,
                scope_in_scope=scope_decision.in_scope,
                scope_similarity=scope_decision.best_similarity,
                bm25_scores=retrieval_result.scores,
                scope_latency_ms=scope_decision.latency_ms,
                retrieval_latency_ms=retrieval_result.latency_ms,
                generation_latency_ms=generation_result.latency_ms,
                total_latency_ms=(time.perf_counter() - total_start) * 1000,
                generator_calls=1,
            )

        if not scope_decision.in_scope:
            return RouteResult(
                decision="out_of_scope",
                answer=config.SCOPE_MESSAGE,
                scope_in_scope=False,
                scope_similarity=scope_decision.best_similarity,
                bm25_scores=retrieval_result.scores,
                scope_latency_ms=scope_decision.latency_ms,
                retrieval_latency_ms=retrieval_result.latency_ms,
                total_latency_ms=(time.perf_counter() - total_start) * 1000,
                generator_calls=0,
            )

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