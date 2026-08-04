# CLAUDE.md

## What this repo is

Undergraduate Computer Engineering thesis: **Optimizing Local Small Language Models Using 4-Bit Quantization, QLoRA, and RAG for University Helpdesks**. It is a controlled prototype evaluation, not a production system. `THESIS_BATTLE_PLAN.md` is the governing document — it overrides any conflicting detail in other markdown notes (`README.md`, `ArchitectureAndDesign.md`, `prompt_and_routing_architecture.md`, `ISO_STANDARDS_AND_THESIS_CONTEXT.md`, `consultation/*`).

## Locked study decisions (do not silently deviate)

- One selected locally hosted SLM — final model not yet fixed.
- English queries only; retrieval is hybrid (BM25 lexical + dense embedding with RRF fusion) — the overall system is hybrid-RAG: both retrievers fetch passage candidates, RRF reranks them, a disjunctive sufficiency gate admits answers when either component's evidence exceeds its threshold, and the SLM generates a cited answer.
- Two routing outcomes only: supported cited RAG answer, or predefined non-answer/office referral. No clarification route.
- QLoRA adapts response *behavior* (citation formatting, conciseness, non-answer handling) — never trains in university facts. Facts live only in the LMS corpus.
- Every comparison (base vs QLoRA, 4-bit vs higher-precision reference) must hold corpus, prompt, query set, decoding settings, and hardware constant.
- Test environment: 8 GB VRAM.

## Architecture

Two-stage pipeline, currently developed as a routing proof-of-concept ahead of the real local-SLM integration:

```
PDFs (rag_pipeline/data/raw/)
  -> rag_pipeline/data_cleaning.py   (Stage 1+2: layout-aware extraction + cleaning)
  -> rag_pipeline/data/sanitize/*.jsonl   (one cleaned JSONL per source PDF)
  -> rag_pipeline/chunk_documents.py   (Stage 3, partial: stitch-then-split chunking)
  -> rag_pipeline/data/chunks/*.jsonl   (one chunked JSONL per source PDF)
  -> [not yet wired: loading chunks into routing/data/corpus.json]

English query -> routing/router.py orchestrates three gates:
  Gate 1  routing/scope_gate.py   embedding-similarity scope check (no LLM call)
  Gate 2  routing/retriever.py    BM25 retrieval + evidence-score threshold (no LLM call)
  Gate 3  routing/generator.py    single cited RAG generation call (the only LLM call)
```

- `routing/config.py` centralizes all tunables (model names, thresholds, paths, prompt contract). Change behavior here, not by hardcoding in the gates.
- Gate 1 (`routing/scope_gate.py`) is NOT a trained classifier — deliberately, to avoid adding a training/evaluation burden to the thesis. It does a cosine-similarity nearest-neighbor lookup against labeled exemplar queries using a frozen pretrained embedding model (`bge-small-en-v1.5`, 512-token max). Gate 2 (dense retrieval) uses the same model so the query is embedded once and reused. No gradient update happens. Keep it this way — do not introduce a trained classifier here.
- Gate 2 (`routing/retriever.py`) now orchestrates both BM25 (lexical) and dense (embedding-based) retrieval, fuses their rankings via RRF, and checks sufficiency as a disjunction: lexical_ok OR dense_ok, where each has its own calibrated threshold. Refer only when both retrievers fail their component gates.
- `SCOPE_SIMILARITY_THRESHOLD`, `BM25_COVERAGE_THRESHOLD`, `DENSE_SCORE_THRESHOLD`, and `RRF_K` in `routing/config.py` are placeholder values — must be calibrated against a labeled eval set before results are meaningful.

## Commands

```bash
# Data cleaning + chunking (run from repo root or rag_pipeline/, paths are self-relative)
python rag_pipeline/data_cleaning.py
python rag_pipeline/chunk_documents.py

# Routing PoC
cd routing
pip install -r requirements.txt
cp .env.example .env   # then paste OPENAI_API_KEY
python main.py --demo   # fixed sample queries
python main.py          # interactive loop
```

There is no configured test runner, linter, or build step in this repo yet (`rag_pipeline/test_chunk_documents.py` exists but is run directly).

## Working conventions

- Hybrid retrieval (BM25 + dense embedding, RRF fusion, disjunctive component gate) is in scope. Do not add reranking beyond RRF, multilingual support, live LMS integration, personal-record handling, or a trained classifier for Gate 1 — explicitly out of scope per `THESIS_BATTLE_PLAN.md`.
- Prompt contract and output JSON shapes (`decision`/`answer`/`citations`) are fixed by `prompt_and_routing_architecture.md`; keep `routing/config.py`'s `SYSTEM_PROMPT` in sync with it rather than duplicating logic elsewhere.
- Any cited identifier in a generated answer must be validated against evidence actually supplied to the model before being treated as trustworthy output. The `document_id` field must be fixed-width (prefix-safe) so substring checks never cross document boundaries.
