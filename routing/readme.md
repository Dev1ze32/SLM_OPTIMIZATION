# Routing PoC — University Helpdesk

Tests whether the two-gate routing design (scope -> evidence -> generation)
works without a trained classifier, per `THESIS_BATTLE_PLAN.md`.

## Module map (read in this order)

| File | Gate | What it does | Touches the LLM? |
|---|---|---|---|
| `config.py` | — | All tunable settings in one place | no |
| `scope_gate.py` | Gate 1 | Embedding-similarity scope check | no (small embedding model only) |
| `retriever.py` | Gate 2 | BM25 retrieval + evidence threshold | no |
| `generator.py` | Gate 3 | Single cited RAG answer call | **yes — the only LLM call** |
| `router.py` | orchestrator | Wires the three gates, tracks latency per gate | — |
| `main.py` | entry point | CLI to run/inspect routing decisions | — |

Each gate is a standalone class with one public method (`check`/`retrieve`/
`generate`), so you can review, test, or swap any one of them without
touching the others.

## Setup

```bash
cd helpdesk_poc
pip install -r requirements.txt
cp .env.example .env
# then edit .env and paste in your real key
```

`config.py` loads `.env` automatically via `python-dotenv` on import — no
need to `export` anything in your shell. `.env` is already listed in
`.gitignore` so the real key never gets committed. If the key is missing
when `RAGGenerator` is constructed, you'll get a clear `RuntimeError`
pointing back here instead of a raw SDK stack trace.

## Run

```bash
python main.py --demo      # fixed sample queries, no typing needed
python main.py             # interactive loop
```

Each result prints the routing decision, the answer, and a **latency
breakdown per gate** (`scope_latency_ms`, `retrieval_latency_ms`,
`generation_latency_ms`) plus `generator_calls` (0 or 1). Keeping these
separate is deliberate — it's what lets you later attribute latency
changes to the generation model alone when you run the base-vs-QLoRA and
4-bit-vs-higher-precision comparisons, without routing overhead
contaminating the numbers.

## What to check while reviewing

- **`scope_gate.py`**: swap `data/scope_exemplars.json` or
  `config.SCOPE_SIMILARITY_THRESHOLD` to see how the scope boundary
  shifts. `best_match_query` on each `ScopeDecision` tells you *why* a
  query was accepted/rejected — useful for threshold tuning against your
  labeled eval set.
- **`retriever.py`**: `config.BM25_SCORE_THRESHOLD` controls the
  evidence-sufficiency cutoff. Try the parking-permit query in
  `main.py`'s sample list — it's deliberately absent from
  `data/corpus.json` to demonstrate the referral path.
- **`router.py`**: confirms the routing-table contract from
  `prompt_and_routing_architecture.md` — `generator_calls` should be `0`
  for `out_of_scope`/`refer` and `1` for `answer`.

## Swapping in the local SLM later

Only `generator.py` needs to change. Replace the `OpenAI` client call
in `RAGGenerator.generate()` with a call to your local 4-bit
Llama 3.2 3B / Qwen3 4B inference endpoint, keep the same
`GenerationResult` return shape, and `router.py`/`main.py` need no edits.

## Known gap to close before real evaluation

Both `SCOPE_SIMILARITY_THRESHOLD` and `BM25_SCORE_THRESHOLD` are
placeholder values (`0.50` and `1.0`). Calibrate them against your
labeled English evaluation set (in-scope-with-evidence,
in-scope-without-evidence, out-of-scope) before drawing any conclusions
about routing correctness.