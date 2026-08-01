"""
Gate 3: RAG answer generation.

Wraps the single generator call described in prompt_and_routing_architecture.md.
This is the ONLY gate that spends a call on the answer-generation model.
Swap config.GENERATOR_MODEL (and, later, the client used here) to point
at a local 4-bit Llama 3.2 3B / Qwen3 4B endpoint without touching
router.py or the other gates.
"""

import time
from dataclasses import dataclass
from typing import List

from openai import OpenAI

import config
from retriever import Passage


@dataclass
class GenerationResult:
    answer: str
    citations: List[str]
    latency_ms: float


def _format_evidence(passages: List[Passage]) -> str:
    blocks = []
    for p in passages:
        blocks.append(
            f"[{p.document_id}] {p.title} ({p.source}, {p.page_or_section}, "
            f"rev. {p.revision_date}):\n{p.text}"
        )
    return "\n\n".join(blocks)


class RAGGenerator:
    def __init__(self, model: str = config.GENERATOR_MODEL):
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Create a .env file in the project "
                "root (see .env.example) or export it in your shell."
            )
        self.model = model
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(self, query: str, passages: List[Passage]) -> GenerationResult:
        start = time.perf_counter()

        evidence_block = _format_evidence(passages)
        user_prompt = (
            f"Evidence:\n{evidence_block}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the evidence above. Cite document id(s) used."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=config.GENERATOR_TEMPERATURE,
            max_tokens=config.GENERATOR_MAX_TOKENS,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer_text = response.choices[0].message.content
        cited_ids = [p.document_id for p in passages if p.document_id in answer_text]

        latency_ms = (time.perf_counter() - start) * 1000

        return GenerationResult(
            answer=answer_text, citations=cited_ids, latency_ms=latency_ms
        )