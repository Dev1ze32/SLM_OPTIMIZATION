"""
Gate 1: Scope gate.

Decides whether an incoming query belongs to the university-helpdesk
domain at all, BEFORE retrieval runs. This is intentionally independent
of BM25 so that weak/failed retrieval is never mistaken for an
out-of-scope query (see prompt_and_routing_architecture.md: "Do not
classify a query as casual conversation merely because BM25 retrieves
weak results.").

Implementation: cosine-similarity nearest-neighbor lookup against a
small set of labeled in-scope exemplar queries, using a pretrained
sentence-embedding model. No classifier is trained and no gradient
update happens here — this is the "without a complex classifier"
approach.

Cost profile: single small forward pass through a lightweight embedding
model (tens of MB, comfortably CPU-capable). It does NOT touch the SLM,
so it never stacks onto generation latency the way an LLM-based scope
prompt would.

Device: pinned to CPU (config.EMBEDDING_DEVICE), deliberately. The
deployment target is a 4-bit quantized SLM on 8 GB VRAM and the SLM needs
all of it; any VRAM this gate allocates competes directly with generation.
sentence-transformers auto-selects cuda when no device is passed, so the
device argument below is load-bearing, not decoration.
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import config


@dataclass
class ScopeDecision:
    in_scope: bool
    best_similarity: float
    best_match_query: str
    latency_ms: float


class ScopeGate:
    def __init__(
        self,
        exemplars_path: str = config.SCOPE_EXEMPLARS_PATH,
        model_name: str = config.EMBEDDING_MODEL,
        threshold: float = config.SCOPE_SIMILARITY_THRESHOLD,
        device: str = config.EMBEDDING_DEVICE,
        cpu_threads: int = config.EMBEDDING_CPU_THREADS,
    ):
        load_start = time.perf_counter()

        self.threshold = threshold
        self.device = device
        if device == "cpu" and cpu_threads:
            torch.set_num_threads(cpu_threads)
        self.model = SentenceTransformer(model_name, device=device)

        with open(exemplars_path, "r", encoding="utf-8") as f:
            self.exemplars = json.load(f)["in_scope_examples"]

        # Pre-embed exemplars once at startup, not per query.
        self.exemplar_embeddings = self.model.encode(
            self.exemplars, normalize_embeddings=True, device=self.device
        )

        self.load_latency_ms = (time.perf_counter() - load_start) * 1000

    def check(self, query: str) -> ScopeDecision:
        start = time.perf_counter()

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, device=self.device
        )[0]
        similarities = self.exemplar_embeddings @ query_embedding

        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_match_query = self.exemplars[best_idx]

        latency_ms = (time.perf_counter() - start) * 1000

        return ScopeDecision(
            in_scope=best_similarity >= self.threshold,
            best_similarity=best_similarity,
            best_match_query=best_match_query,
            latency_ms=latency_ms,
        )