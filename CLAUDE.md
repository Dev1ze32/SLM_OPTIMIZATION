# CLAUDE.md

## What this repo is

Undergraduate CpE thesis: **Optimizing Local Small Language Models Using 4-Bit Quantization, QLoRA, and RAG for University Helpdesks**. Controlled prototype evaluation, not production. Currently in **proposal phase** (Chapters 1–3), building this routing PoC ahead of the real prototype.

`THESIS_BATTLE_PLAN.md` governs; it overrides `README.md`, `ArchitectureAndDesign.md`, `prompt_and_routing_architecture.md`, `EVALUATION_PLAN.md`, `ISO_STANDARDS_AND_THESIS_CONTEXT.md`.

`consultation/*` is **superseded** (an earlier, larger scope: clarification route, multilingual eval, reranking, Q2 adapter, concurrency testing, prompt variants). Design rationale only — don't cite it or reintroduce its scope; `EVALUATION_PLAN.md` lists each exclusion explicitly.

## Locked study decisions (do not silently deviate)

- One locally hosted SLM, final model not yet fixed.
- English queries only. Retrieval is hybrid-RAG: BM25 (lexical) + dense embedding fetch candidates, RRF fuses rankings, a disjunctive sufficiency gate admits answers when either component exceeds its own threshold, then the SLM generates one cited answer.
- Two routing outcomes only: cited RAG answer, or predefined non-answer/office referral. No clarification route.
- QLoRA adapts response *behavior* (citation formatting, conciseness, non-answer handling) — never trains in facts. Facts live only in the LMS corpus.
- Every comparison (base vs QLoRA, 4-bit vs higher-precision reference) holds corpus, prompt, query set, decoding settings, and hardware constant.
- Test environment: 8 GB VRAM.
- Hybrid is justified by this corpus specifically: form codes (`PNC:AA-FO-45`), acronyms, and figures favor BM25; student phrasing vs. policy register ("can I take a break" → "Leave of Absence") favors dense. Neither alone covers both — this is the Chapter 3 justification, keep it.

## Architecture

```
PDFs (rag_pipeline/data/raw/)
  -> data_cleaning.py -> data/sanitize/*.jsonl   [tables: see docs/TABLE_HANDLING_PLAN.md;
                                                   ambiguous table regions logged to
                                                   data/sanitize/table_report.json for review]
  -> chunk_documents.py -> data/chunks/*.jsonl   [done: 364 chunks total, 22/234/108]
  -> build_corpus.py -> routing/data/corpus.json [done: 364 real passages, replaces the old 5-entry placeholder]

English query -> routing/router.py orchestrates three gates, retrieval before scope:
  Gate 2  retriever.py    BM25 done; dense + RRF + disjunctive gate NOT YET IMPLEMENTED (BM25 only today)
  Gate 1  scope_gate.py   embedding cosine-similarity vs labeled exemplars (no LLM, no training) —
                          runs every query, but only consulted when Gate 2 finds insufficient evidence
  Gate 3  generator.py    single cited RAG call — currently gpt-4o-mini, swap for local SLM before any reported result
```

- `routing/config.py` centralizes all tunables — change behavior there, not by hardcoding in the gates.
- Gate 1 and Gate 2's dense half share one embedding model (`bge-small-en-v1.5`), embedded once per query. Gate 1 stays a frozen nearest-neighbor lookup — do not introduce a trained classifier.
- No vector database. Chunk embeddings live in an in-memory NumPy array (same pattern as `scope_gate.py`), cached to `.npy` so they aren't recomputed each run. Exhaustive cosine similarity over 364 chunks is exact and sub-millisecond — do not add Chroma/FAISS/Pinecone.
- `SCOPE_SIMILARITY_THRESHOLD`, `BM25_COVERAGE_THRESHOLD`, `DENSE_SCORE_THRESHOLD`, `RRF_K` are placeholders — calibrate against a labeled eval set.
- Resolved: Gate 1 runs after Gate 2, not before, and only decides the *reason* for a non-answer (out-of-scope vs. referral) when retrieval alone is insufficient. This removes the false-out-of-scope failure mode (a paraphrased in-scope query rejected before retrieval could match it) at no extra cost — retrieval over 364 chunks is milliseconds against the generator's latency. `scope_similarity` is still logged on every query, including the answer path, so a false-rejection rate under the old ordering can be reported from that log.

## Evaluation

Full design lives in `EVALUATION_PLAN.md` — configurations, query-set strata, pooling protocol, effort budget, triage order. Facts that constrain code:

- Three configs (A=base, B=QLoRA, C=higher-precision), all at hybrid retrieval, one-factor-at-a-time from shared anchor A — not a factorial grid.
- Retrieve once per query, cache, replay identical evidence across A/B/C.
- Report Recall@k for **k ≤ 10 only** (the pooling protocol only covers that far).

## Commands

```bash
python rag_pipeline/data_cleaning.py
python rag_pipeline/chunk_documents.py
python rag_pipeline/build_corpus.py   # writes routing/data/corpus.json from the real chunks

cd routing
pip install -r requirements.txt
cp .env.example .env   # paste OPENAI_API_KEY
python main.py --demo   # fixed sample queries
python main.py          # interactive loop
```

No configured test runner, linter, or build step (`rag_pipeline/test_chunk_documents.py` is run directly).

`rag_pipeline`'s dependencies (`fitz`/PyMuPDF, etc.) are installed in a WSL venv at `venv/` (Python 3.13), not the Windows-side Python. Run pipeline scripts through it: `wsl.exe -e bash -lc "cd /mnt/c/.../rag_pipeline && ../venv/bin/python data_cleaning.py"`. Same for `routing/`'s dependencies (`openai`, `rank_bm25`, `sentence_transformers`, `torch`) — that venv has all of them installed already.

## Working conventions

- Do not add reranking beyond RRF, multilingual support, live LMS integration, or personal-record handling — out of scope per `THESIS_BATTLE_PLAN.md`.
- Keep `routing/config.py`'s `SYSTEM_PROMPT` in sync with `prompt_and_routing_architecture.md` rather than duplicating the contract elsewhere.
- Validate every cited identifier against evidence actually supplied to the model before treating it as trustworthy. `document_id` must be fixed-width (prefix-safe) so substring checks never cross document boundaries.
