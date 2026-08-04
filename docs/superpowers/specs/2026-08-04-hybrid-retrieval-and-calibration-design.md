# Hybrid Retrieval, Corpus Build, and Threshold Calibration — Design

Date: 2026-08-04
Status: awaiting approval
Scope: `rag_pipeline/data/chunks/*.jsonl` → queryable retrieval layer for `routing/`

---

## 0. Scope change notice (read first)

This design **deviates from a locked study decision** and cannot be implemented until that
is resolved on the thesis side.

| Document | Line | Current text |
|---|---|---|
| `THESIS_BATTLE_PLAN.md` | 14 | "BM25 lexical retrieval is the only retrieval method in the prototype." |
| `THESIS_BATTLE_PLAN.md` | 60 | Excluded: "...dense/hybrid retrieval, reranking..." |
| `THESIS_BATTLE_PLAN.md` | 75 | "report this as a retrieval limitation rather than adding dense retrieval **without approval**" |
| `CLAUDE.md` | — | "retrieval is BM25 lexical only (no dense/hybrid retrieval, no reranking, no RRF)" |

The user has directed that hybrid retrieval be added. Line 75 frames dense retrieval as
permitted *with* approval, so this is an approved scope change rather than a violation —
but two things follow:

1. **Adviser sign-off is required and is the user's to obtain.** This plan assumes it.
2. **`THESIS_BATTLE_PLAN.md` and `CLAUDE.md` must be updated** as part of implementation.
   Leaving them contradicting the code would leave the methodology chapter describing a
   system that no longer exists. This is a tracked work item, not an afterthought.

**Evaluation cost of the change:** the study must now demonstrate that hybrid beats BM25
alone. That means running the labelled eval set through **three** configurations
(BM25-only, dense-only, hybrid) instead of one, and the eval set must contain query
categories that can distinguish them (§5.2). This is genuine additional thesis work. It is
also a defensible contribution, since the comparison is exactly what justifies the change.

---

## 1. Findings from the chunk review

Read-only inspection of all 364 chunks across the three manuals. Nothing was modified.

### 1.1 Inventory

| Source | Chunks | median chars | p90 | max |
|---|---|---|---|---|
| `UC-PnC-Student-Handbook-Manual` | 234 | 1614 | 1790 | 2408 |
| `UC-PnC-Student-Research-and-Innovation-Manual` | 108 | 1635 | 1795 | 4095 |
| `UC-PnC-Internship-Manual` | 22 | 1550 | 1752 | 1795 |
| **Total** | **364** | — | — | — |

Corpus size: **502,498 chars ≈ 126K tokens**. This is a small corpus, and that fact drives
most of the engineering decisions below.

### 1.2 Format consistency — clean

All 16 fields present in all 364 records. No schema drift, no missing keys, no type
inconsistency. `chunk_documents.py` produces well-formed output.

```
source_file, pdf_page_start, side_start, pdf_page_end, side_end,
printed_page_start, printed_page_end, pages_spanned, stitched,
page_refs, split_index, split_count, section_path, char_count, text, tables
```

`printed_page_start` is non-null on all 364 records — citations can always cite a printed
page number. 110 chunks are stitched across pages; 242 came from a size-cap split; 27
carry tables.

### 1.3 Issues found

| Issue | Count | Impact |
|---|---|---|
| **Exact duplicate texts** | 3 texts × 3 copies = 6 redundant | Institutional boilerplate (Vision, Mission/LGU linkage, Graduate Attributes) appears identically in all three manuals. At `top_k=3` a boilerplate hit can fill **every** evidence slot with the same text. |
| **Empty `section_path`** | 21 | No breadcrumb prepended → weaker lexical signal and a citation with no section. Concentrated in front-matter-adjacent content pages. |
| **Over the 1800-char cap** | 19 (max 4095) | 5 worst are one table series in the Research manual; a single such chunk is ~1000 tokens of prefill. |
| **Very short chunks** | 8 under 200 chars | Mostly single outline items under a long breadcrumb; low standalone information content. |

No malformed records, no encoding damage, no empty `text`.

### 1.4 No stable chunk identifier exists

There is no `chunk_id` field. `(source_file, page_refs, split_index)` is verified unique
across all 364 records and is the natural key (§3.2).

### 1.5 The corpus the router actually loads is a toy

`routing/data/corpus.json` currently holds **5 hand-written passages** of ~220 chars each,
with invented `document_id`s (`LMS_001`) and generic US-university content ("financial
aid", "add/drop period"). The real chunks are 73× more numerous and ~7× longer each.

**Both thresholds in `config.py` were set against this toy corpus and do not transfer.**

### 1.6 Measured: the evidence gate is currently a no-op

BM25 top-1 scores over the *real* 364 chunks, using `retriever.py`'s exact tokenizer
(`text.lower().split()`) and a faithful reimplementation of `BM25Okapi`:

| Query set | top-1 score range |
|---|---|
| 8 in-scope UC-PnC queries | **9.80 – 23.07** |
| 4 out-of-scope junk queries | **4.65 – 11.02** |

`BM25_SCORE_THRESHOLD = 1.0` admits **everything**. `"How do I cook chicken adobo?"`
scores 6.27 and passes. The `refer` route — one of the two locked routing outcomes — never
fires on retrieval grounds against the real corpus.

Worse, the two ranges **overlap**: `"What is the weather in Manila today?"` (11.02)
outscores a legitimate in-scope query (9.80). No absolute threshold can separate them as
currently tokenized.

### 1.7 Measured: the tokenizer is the root cause

`_tokenize` is `text.lower().split()` — no punctuation stripping, no stemming, no
stopwords. It indexes **4,285 punctuation-glued surface forms** (`'20.'`, `'2003,'`,
`'2003-059,'`), so `policy` and `policy.` are distinct terms.

| | current | regex + stopwords + stemming |
|---|---|---|
| unique indexed terms | 9,377 | 4,629 |
| avg doc length (BM25 normalizes by this) | 208 tok | 134 tok |
| in-scope top-1 range | 9.80 – 23.07 | 6.56 – 19.45 |
| out-of-scope top-1 range | 4.65 – **11.02** | 4.67 – **6.27** |
| **separable?** | **no (overlap)** | **yes (6.27 < 6.56)** |

Top-1 also changed on 2 of 8 queries, both toward better targets (grievance → Code of
Conduct chapter; IP → Chapter V Intellectual Property Rights).

> **Caveat, stated plainly:** 8 in-scope + 4 out-of-scope queries written by the assistant,
> scored with a light demonstration stemmer. The separation margin (6.27 vs 6.56) is thin.
> This is a directional signal that motivates the calibration work in §5 — **not** a
> calibrated result, and it must not be reported as one.

### 1.8 Measured: `all-MiniLM-L6-v2` cannot embed these chunks

The model Gate 1 already uses caps at **256 wordpiece tokens**.

| Model limit | Chunks truncated | Corpus text dropped |
|---|---|---|
| **256** (MiniLM, current) | **248–266 of 364 (68–73%)** | **~20%** |
| **512** (bge-small / e5-small / gte-small) | 1–6 of 364 (0–2%) | ~0% |

*(Estimated at 1.30–1.45 tokens/word. The gap is too wide for the estimate to flip it, but
the exact figure should be re-measured with the real tokenizer at implementation time.)*

Embedding chunks with MiniLM would silently drop ~a fifth of the corpus — specifically the
*tail* of every long chunk, which is often where the policy detail lives. It would present
as "dense retrieval underperforms."

### 1.9 Latent bug in citation validation

`generator.py:70` does `p.document_id in answer_text` — a **substring** test. Safe for the
toy `LMS_001` ids, but any variable-width scheme (`HB_1`, `HB_10`, `HB_100`) makes `HB_1`
match a citation of `HB_100`. **Document ids must be fixed-width to be prefix-safe.**

---

## 2. Retrieval architecture

```
query
  ├─ embed ONCE (bge-small-en-v1.5) ──────────────┐
  │                                               │
  ├─ Gate 1  scope: cosine vs exemplar queries ◄──┤   (reuses the same vector)
  │                                               │
  └─ Gate 2  evidence:                            │
       ├─ BM25 (rank_bm25) ────► lexical top-N    │
       ├─ dense (numpy dot) ◄───────────────────  ┘  dense top-N
       │
       ├─ SUFFICIENCY: disjunctive component gate
       │     lexical_ok = coverage ≥ c  OR  norm_bm25 ≥ s
       │     dense_ok   = max_cosine ≥ d
       │     sufficient = lexical_ok OR dense_ok
       │
       ├─ FUSION: RRF over ONLY the lists whose gate fired
       │
       └─ top_k passages ─► Gate 3 generation (the only LLM call)
```

### 2.1 Separation of concerns

> **Fusion decides order. The gate decides whether to answer. They read different signals.**

This separation is forced, not stylistic. **RRF is rank-only**: its score is
`Σ 1/(k + rank)`, so the top result for `"best pizza recipe"` receives the *identical*
fused score as the top result for a perfect query — both are rank 1 in both lists.
Thresholding the fused score would destroy the referral route entirely, which is strictly
worse than today's broken-but-scored gate. **Sufficiency must therefore be judged on
component scores, before fusion.**

### 2.2 Disjunctive component gate

`sufficient = lexical_ok OR dense_ok`, each with its own calibrated threshold.

Rationale: the two retrievers have *different* failure modes. BM25 misses paraphrase; dense
misses rare exact tokens (`R.A. 10627`, form codes, section numbers, proper nouns). A
disjunction refers only when **both** agree there is nothing — which is the actual
definition of insufficient evidence.

This also resolves the concern that a coverage gate rejects valid paraphrases. In a
BM25-only design coverage was doing two jobs — precision mechanism *and* sole gate — which
is what made it dangerous. Here it is one of two disjunctive paths, so a paraphrase that
fails coverage still passes on the dense signal. **Coverage becomes safe precisely because
it stops being load-bearing alone.**

Accepted cost: disjunction is recall-oriented, so either retriever alone can admit a weak
match. Backstopped by the prompt contract (`SYSTEM_PROMPT` forbids unsupported answers) and
by citation validation (§3.4). Rejected alternatives: **conjunctive** (refers on
keyword-only *and* paraphrase queries — cancels the benefit of hybrid) and **dense-only
gate** (loses the exact-token safety net).

### 2.3 What happens when the retrievers disagree

**Gate disagreement** — fuse only over the retrievers that *passed* their own gate:

| Situation | Evidence sent to the generator |
|---|---|
| Both fire | RRF over both lists |
| Only dense fires | Dense ranking alone (paraphrase/intent query) |
| Only BM25 fires | BM25 ranking alone (exact-token query) |
| Neither fires | `refer` — no generator call |

If a retriever's gate did not fire, its candidates are best-of-a-bad-lot. Including them
would add noise to the evidence block and burn prefill tokens for nothing — against both
the accuracy and the performance goal.

**Ranking disagreement** — this is what RRF is for. Its defining property is that it
rewards *agreement* over a single strong opinion: a chunk ranked 3rd by both beats a chunk
ranked 1st by one and 8th by the other. How strongly is controlled by `k`, and the
conventional default is wrong at this scale:

| `k` | rank-1 vs rank-10 score spread | behavior |
|---|---|---|
| 60 (paper default) | **1.15× — nearly flat** | fusion barely reorders anything |
| 10 | 1.82× | agreement wins, with real separation |
| 1 | 5.50× | a strong single #1 beats consensus |

`k=60` was tuned for TREC runs over thousands of candidates. Fusing ~10 candidates it is
close to inert. **`RRF_K` is a calibration parameter in `config.py`, swept with the
thresholds (§5.3) — not a copied constant.**

**Disagreement as a metric.** Log per-query rank overlap between the two retrievers. If
they always agree, hybrid adds nothing and the thesis should report that honestly; if they
disagree and fusion wins, that is the empirical justification for the scope change in §0.

### 2.4 Retained from the BM25-only plan

- **Tokenizer fix** — regex tokenization + stopword removal + Snowball stemming, applied
  identically at index and query time. §1.7. Adds an `nltk` dependency or a vendored stemmer.
- **Dedupe** — collapse the 3 boilerplate texts to one passage each, **carrying all three
  source references** rather than dropping copies, so a citation can name every manual the
  text appears in. Lossless, and frees the wasted `top_k` slots.
- **Eval scaffold** — §5.

**Deferred:** BM25 `b`/`k1` tuning. Meaningless before calibration data exists; revisit
after §5.3.

---

## 3. Storage schema

### 3.1 No vector database

**364 chunks × 384 dims × 4 bytes = 546 KB.**

A single `(364×384) @ (384,)` matvec is **279,552 FLOPs** — it fits in L2 cache and runs in
microseconds on CPU. ANN indexes (HNSW, IVF) exist to *avoid* exhaustive search over ~10⁶⁺
vectors by trading accuracy for speed. At 364 vectors, exhaustive search is both **faster
and exact**. Given the stated goal — performance without losing accuracy — it strictly
dominates. Chroma / Qdrant / pgvector would each add a service to run, a dependency to
reproduce, and a slower answer.

Embedding is done by a **local** model, so build cost is $0 and no network call is
involved — which also preserves the battle plan's "offline prototype" claim.

### 3.2 Artifacts

```
routing/data/corpus.json      passages, row i
routing/data/embeddings.npy   float32 (364, 384), L2-normalized AT BUILD TIME
routing/data/manifest.json    model id, dim, corpus hash, chunker config, source hashes
```

Pre-normalizing at build time turns query-time cosine into a plain dot product — one less
pass per query.

**Passage schema.** The current 6-field shape is lossy: it has one `page_or_section` string,
but chunks carry page *spans* and a breadcrumb. Battle plan step 2 requires "stable citation
metadata" and `CLAUDE.md` requires validating cited identifiers against supplied evidence —
neither works against a squashed string. Proposal: extend `Passage` with structured fields
and keep `page_or_section` as a **derived display string**, so the prompt format is
unchanged.

| Field | Source | Note |
|---|---|---|
| `document_id` | generated | `HB_001` / `RI_001` / `IN_001` — **fixed-width, prefix-safe** (§1.9) |
| `title`, `source` | per-document map | **user-supplied — decision D1** |
| `revision_date` | per-document map | **user-supplied — chunks carry no date — decision D1** |
| `page_or_section` | derived | e.g. `"Chapter V > Grading System, pp. 39-40"` |
| `text` | `chunk.text` | breadcrumb already prepended by the chunker |
| `section_path`, `source_file`, `page_refs`, `printed_page_start/end` | passthrough | enables programmatic citation validation |

Touches `retriever.py`, `generator.py`, `corpus.json`. No logic changes.

### 3.3 Identifier stability

`(source_file, page_refs, split_index)` is unique across all 364 records and is
citation-meaningful and debuggable. A content hash would be stable under *reordering* but
not under re-chunking either, since the text itself changes.

**Neither scheme survives a chunker change**, and that is the real constraint:

> **Freeze `CHUNK_SIZE_CAP` and `chunk_documents.py` before the eval set is labelled.**
> Gold labels reference chunk ids; re-chunking silently invalidates them.

### 3.4 Citation validation

Replace the substring test at `generator.py:70` with an exact match against the
`document_id`s actually supplied in the evidence block, and additionally verify that any
page or section cited in the answer text is consistent with the structured fields of the
cited passage. Per `CLAUDE.md`, no cited identifier is trustworthy until validated against
evidence actually supplied.

---

## 4. Embedding approach

| Decision | Value | Why |
|---|---|---|
| Model | **`bge-small-en-v1.5`** | 512-token limit → ~0% truncation vs MiniLM's 68–73% (§1.8). 33M params, 384d, strong CPU retrieval quality. |
| Gate 1 model | **same model** | One model in memory (~130 MB CPU RAM, **zero VRAM**), and the query is embedded **once and reused** by both gates. |
| Device | CPU | `EMBEDDING_DEVICE = "cpu"` already set; the 4-bit SLM needs all 8 GB of VRAM. |
| Batching | build-time only, batch 32 | 364 chunks is a one-shot local job of seconds. No streaming, no rate limits, no API. |
| Query-time cost | **~0 added** | Gate 1 already pays the embedding; dense retrieval adds only the matvec (§3.1). |
| Cost | **$0** | Local model. No embedding API. |

`bge` models expect a query-side instruction prefix for asymmetric retrieval
(`"Represent this sentence for searching relevant passages: "`) applied to the **query
only**, not to passages. Getting this wrong degrades retrieval quietly — it must be
covered by a test.

**Changing Gate 1's model costs nothing**, because `SCOPE_SIMILARITY_THRESHOLD = 0.50` is
an uncalibrated placeholder that is being recalibrated anyway (§5.3).

### 4.1 Chunk-length risk

At 512 tokens only 1–6 chunks truncate, so per-chunk embedding is used directly. If the
real tokenizer shows materially more truncation, the fallback is **sub-chunk embedding**:
split over-length chunks, embed each piece, score the parent by max sub-vector similarity.
This keeps the chunk as the citation unit and costs ~370 vectors / 555 KB. Not adopted now
— recorded so the decision is not re-derived later.

---

## 5. Evaluation set and calibration

Nothing in §2 can be tuned without labels. `THESIS_BATTLE_PLAN.md:64` already makes this
step 1. No eval set exists anywhere in the repo.

### 5.1 Scaffold (assistant) → labels (user)

`rag_pipeline/build_eval_template.py` emits a pre-filled CSV: a stratified sample across all
three manuals and section types, one row per sampled chunk, with empty `query` /
`expected_decision` slots. The user authors the queries and ground truth — the labels must
be domain-authored to be defensible in the writeup.

### 5.2 Required query categories

Hybrid must be *shown* to beat BM25 alone, so the set must be able to distinguish them:

| Category | Target | Purpose |
|---|---|---|
| Exact-token / keyword | ~15 | BM25 should win (`R.A. 10627`, form codes, section numbers) |
| Paraphrase / intent | ~15 | Dense should win — **this is what justifies the scope change** |
| Plain factual | ~20 | Both should succeed |
| In-scope but absent | ~15 | Must produce `refer` |
| Out-of-scope | ~10 | Must produce `out_of_scope` at Gate 1 |

### 5.3 Calibration sweep

Sweep and report the precision/recall trade-off so the user picks the operating point:

- `SCOPE_SIMILARITY_THRESHOLD` (Gate 1, cosine)
- `DENSE_SCORE_THRESHOLD` (`d`)
- `BM25_COVERAGE_THRESHOLD` (`c`) and `BM25_SCORE_THRESHOLD` (`s`, normalized)
- `RRF_K` — §2.3
- `BM25_TOP_K` / candidate pool `N`

Reported per configuration (**BM25-only / dense-only / hybrid**): retrieval recall@k,
answer/refer confusion matrix, false-accept rate on out-of-scope, and retriever agreement
rate (§2.3).

---

## 6. Performance budget

Retrieval is not the bottleneck and adding dense retrieval does not make it one.

| Stage | Cost |
|---|---|
| BM25 over 364 chunks | sub-millisecond |
| Dense retrieval | ~0 added (query vector reused; 279K-FLOP matvec) |
| Embedding model resident | ~130 MB **CPU RAM**, zero VRAM |
| Vectors on disk | 546 KB |
| **Evidence prefill at `top_k=3`** | **~1,210 tokens median, ~2,540 worst case** |

**Prefill dominates.** It is controlled by `top_k` and chunk size, not by the retriever, and
is the term to tune for latency on 8 GB. `top_k` and an evidence char cap are therefore
treated as calibrated tunables (§5.3), measured rather than guessed. The five oversized
Research-manual table chunks (up to 4095 chars ≈ 1000 tokens each) are the main worst-case
driver.

`router.py`'s existing per-gate latency fields are kept; dense retrieval time folds into
`retrieval_latency_ms` so generation-only latency stays isolable for the base-vs-QLoRA and
4-bit-vs-reference comparisons.

---

## 7. Re-runs

`corpus.json` and `embeddings.npy` are **pure build artifacts**. Full deterministic rebuild
every time — **no upserts, no persisted ANN index, no migration logic.** Rebuilding 364
chunks is seconds of local CPU; incremental complexity is not justified at this scale.

The one real hazard is that there are now **two artifacts that must stay in lockstep**. If
chunks change and embeddings do not, row *i* silently stops corresponding to passage *i* —
a correctness bug that produces confident, plausible, wrong citations.

Mitigation:

1. **One build script emits both**, never independently.
2. **`manifest.json`** records embedding model id, dim, normalization flag, corpus content
   hash, chunker config, and source file hashes.
3. **The loader asserts the manifest matches and refuses to start on mismatch.** Fail loud,
   never silent.
4. A rebuild that changes the corpus hash **flags the eval set as stale** (§3.3).

---

## 8. Proposed work order

| # | Item | Depends on |
|---|---|---|
| 1 | Adviser sign-off; update `THESIS_BATTLE_PLAN.md` + `CLAUDE.md` | **D0** |
| 2 | Freeze the chunker (§3.3) | — |
| 3 | `build_corpus.py`: chunks → `corpus.json` + dedupe + fixed-width ids | D1, D2, D3 |
| 4 | Tokenizer fix in `retriever.py` | 3 |
| 5 | `build_embeddings.py` → `embeddings.npy` + `manifest.json` | 3 |
| 6 | `DenseRetriever` + RRF fusion + disjunctive gate | 4, 5 |
| 7 | Fix citation validation (§1.9, §3.4) | 3 |
| 8 | `build_eval_template.py` → user labels | 2 |
| 9 | Calibration sweep across 3 configurations | 6, 8 |

Steps 1–2 gate everything. Step 8's labelling is the long pole and can run in parallel with
3–7.

---

## 9. Decisions needed before implementation

| id | Decision | Why it blocks |
|---|---|---|
| **D0** | **Adviser approval for hybrid retrieval**, and confirmation that `THESIS_BATTLE_PLAN.md` / `CLAUDE.md` should be updated to match | §0. Everything else is downstream of the study scope being settled. |
| **D1** | `title`, `source`, and `revision_date` for each of the three manuals | These appear **verbatim in every citation**. Chunks carry no date; there is no defensible value to invent. |
| **D2** | Confirm dedupe treatment: one passage carrying **all three** source references, vs. dropping duplicate copies | Affects whether a citation can name the manual the student actually asked about. |
| **D3** | Confirm extending the `Passage` dataclass with structured citation fields | Alternative is keeping the 6-field shape and accepting citations that cannot be programmatically validated. |
| **D4** | Latency/accuracy operating point — is there a hard per-query latency target on the 8 GB box? | Sets whether `top_k=3` survives calibration or must drop to 2 (§6). |
| **D5** | Who authors the eval queries (§5.1 assumes the user) and what the target size is | Step 8 is the long pole; ~75 queries across 5 categories is the §5.2 proposal. |

### Open tradeoffs flagged for input

- **Disjunctive gate is recall-oriented.** It admits weak matches that a conjunctive gate
  would refer. Accepted deliberately (§2.2); revisit if calibration shows an unacceptable
  false-accept rate.
- **`nltk` dependency** for Snowball stemming, or vendor a stemmer to keep the prototype
  dependency-light.
- **`bge-small` vs `e5-small-v2` vs `gte-small`** — all 512-token, 384d, similar CPU cost.
  `bge-small-en-v1.5` is the recommendation; the build script should treat the model as a
  config-level swap so this can be re-tested cheaply.
- **The §1.7 tokenizer result is directional, not calibrated.** It must not be reported as
  a finding until reproduced against the labelled eval set.
