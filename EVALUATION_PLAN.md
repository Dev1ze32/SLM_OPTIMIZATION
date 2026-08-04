# Evaluation Plan

## Purpose

This document fixes the measurement design for the study described in [THESIS_BATTLE_PLAN.md](THESIS_BATTLE_PLAN.md). It exists to keep the evaluation finishable by one undergraduate author: it states exactly which configurations are run, which numbers are produced automatically versus by human judgment, and what is deliberately not measured.

Every number that appears in Chapter 4 should trace to a row in this document.

## Configurations under test

Comparisons are one-factor-at-a-time from a shared anchor, **not** a factorial. Configuration A appears in both comparisons, so two controlled comparisons cost three configurations.

| ID | Model | Precision | Retrieval | Role |
| --- | --- | --- | --- | --- |
| **A** | base | 4-bit | hybrid | Anchor; shared by both comparisons |
| **B** | QLoRA-adapted | 4-bit | hybrid | A vs B isolates the adapter |
| **C** | base | higher-precision reference | hybrid | A vs C isolates quantization |

- **Comparison 1 (QLoRA):** A vs B. Answer correctness, groundedness, citation accuracy, routing accuracy.
- **Comparison 2 (Quantization):** A vs C. Latency, throughput, peak VRAM, plus groundedness and citation accuracy to detect quality loss.
- **Retrieval ablation:** BM25-only vs dense-only vs hybrid. Retrieval metrics only — no LLM call, no human grading. See below.

Retrieval is held at hybrid for A, B, and C. The ablation never propagates into generation.

## Evaluation query set

Approximately 80 English queries, stratified:

| Stratum | Count | Correct behavior | Needs gold chunks? |
| --- | --- | --- | --- |
| Answerable from corpus | ~40 | cited answer | yes |
| In scope, no corpus evidence | ~25 | referral | no |
| Out of scope | ~15 | scope message | no |

Source queries from real students where possible (student council, guidance office, a short form). Queries authored solely by the researcher against a corpus the researcher also curated is a construct-validity weakness; if self-authored queries are unavoidable, state it as a limitation.

Split by source document and paraphrase family — never randomly — so no QLoRA training question resembles an evaluation question.

## Gold-chunk annotation: pooling protocol

Annotation scales with **queries**, not corpus size. The corpus is 364 chunks; you never label all of it.

1. For each of the ~40 answerable queries, run BM25 top-10 and dense top-10.
2. Take the union and deduplicate — typically 12–20 unique candidates per query.
3. Judge each candidate binary: does it contain the information needed to answer?
4. Cross-check using `section_path` metadata (e.g. `GUIDELINES > CHAPTER II STUDENT ENROLMENT`) to catch relevant chunks the retrievers missed.
5. Freeze the pool before any generation run.

Estimated cost: ~40 queries × ~15 candidates ≈ 600 fast binary judgments ≈ 2–3 hours.

**Constraint:** because the pool is built from BM25 top-10 ∪ dense top-10, and hybrid/RRF can only return documents already present in one of those two lists, all three retrieval configurations are fairly covered — **provided Recall@k is reported for k ≤ 10 only.**

**Limitation to disclose in Chapter 3:** a relevant chunk that no retriever surfaced goes unjudged and is treated as irrelevant, which slightly inflates Recall@k. This is standard pooled-evaluation practice (as used in TREC), but it must be stated.

## Metric cost split

Only two metric families require human judgment. Everything else is a script.

| Metric | Cost | Applies to |
| --- | --- | --- |
| Recall@k (k ≤ 10), MRR, nDCG | automatic | retrieval ablation only |
| Routing decision correct (answer vs refer) | automatic — compare to gold stratum label | A, B, C |
| Citation **validity** (ID ∈ supplied evidence) | automatic — string check | A, B, C |
| Coverage, selective risk | automatic — derived | A, B, C |
| Latency, TTFT, throughput, peak VRAM | automatic | A, B, C |
| Answer **correctness** | **human** | answerable queries only |
| **Groundedness** / citation **entailment** | **human** | answerable queries only |

Routing accuracy over the 40 unanswerable and out-of-scope queries is fully automatic — a confusion matrix against the stratum label. That covers half the query set at zero grading cost.

## Evidence replay

Retrieve once per query using the frozen hybrid configuration, cache the evidence envelope, and replay identical evidence to A, B, and C when grading quality.

This removes retrieval variance as a confound — any observed difference between A and B is attributable to the adapter, not to a different passage being retrieved — and means retrieval runs once rather than three times.

Latency, throughput, and VRAM must still be measured on live end-to-end runs, not replayed ones.

## Run counts and determinism

| Measurement | Runs | Rationale |
| --- | --- | --- |
| Quality metrics (correctness, groundedness, citations, routing) | 1 | Decoding is temperature 0 with a fixed seed; re-running reproduces essentially the same output, so additional runs measure nothing |
| Latency, TTFT, throughput, peak VRAM | ≥ 3, report mean and spread | Genuine run-to-run variance |

Disclose the caveat: temperature 0 reduces sampling variability but is not bit-exact across batching and hardware, and it does not guarantee correctness.

## Human evaluation protocol

- **Volume:** 3 configurations × ~40 answerable queries = ~120 answers to grade (≈ 4 hours with a written rubric).
- **Rubric:** written scoring guide for correctness, groundedness, and citation entailment. Include it as a thesis appendix.
- **Second annotator:** independent grading of a ~30% subset (~36 answers). Report Cohen's κ.
- Sole-annotator grading of one's own system is a credibility problem. Shrink the second-annotator subset before dropping it.

## Explicitly excluded

These appear in the archived `consultation/` notes and must not creep back in.

| Excluded | Source of creep | Reason |
| --- | --- | --- |
| Q2 (behavior + facts QLoRA), stale-fact experiment | `03` | Third adapter arm; already cut |
| Quantization frontier (Q8 / Q5_K_M / Q4_K_M / Q3) | `05` | Two precision points only, per battle plan |
| Prompt variants P0–P3 | `06` | A fourth experimental axis |
| Chunking strategy comparison | `04` | Corpus is already chunked; fix it and justify in prose |
| Embedding model comparison | `04` | `bge-small-en-v1.5` selected on size and CPU constraint |
| Concurrency testing (1/2/4 users) | `05` | Production concern, not a research question |
| Reranking beyond RRF | `02` | Out of scope per battle plan |
| Clarification route | `01`, `02` | Two routing outcomes only |
| Multilingual / Filipino evaluation | `06`, `07` | English only |

## Effort budget

The experiments are not what consumes the schedule — dataset construction is.

| Task | Estimate |
| --- | --- |
| Build and stratify the ~80-query evaluation set | days (depends on student sourcing) |
| Gold-chunk annotation via pooling | 2–3 hours |
| QLoRA behavior training data (~150–300 examples, 5 categories) | days |
| Write automatic scoring scripts | 1–2 days |
| Human grading (120 answers + 36 second-annotator) | ~5 hours |
| Latency / VRAM benchmark runs | hours |

**QLoRA behavior categories** (reduced from the 13 in `consultation/03`): supported single-source answer, supported multi-source answer, insufficient evidence, out-of-scope, malformed citation. Vary the abstention wording — identical refusal text trains an over-refuser, which would silently corrupt the headline A vs B comparison.

Drafting training candidates with a hosted model against the corpus and then hand-reviewing every one is acceptable and saves days. Disclose it in Chapter 3.

## Triage order

If the schedule collapses, drop from the bottom. Decide this now, not in the final month.

1. Routing accuracy, coverage, selective risk — automatic; the spine of the paper
2. A vs B (base vs QLoRA) on answerable queries — the title's core claim
3. A vs C latency / VRAM / throughput — automatic
4. Second-annotator agreement — shrink the subset before dropping
5. Retrieval ablation — first to go; justify hybrid from literature and corpus evidence instead, and list the ablation as future work

The ISO 25010 / 25059 mapping is not on this list: it costs no extra measurement, only a table mapping numbers you already have. See [ISO_STANDARDS_AND_THESIS_CONTEXT.md](ISO_STANDARDS_AND_THESIS_CONTEXT.md).

## Still to be specified

- Final SLM selection (name it in the proposal with feasibility criteria).
- Where QLoRA training runs. Training and inference environments may differ; state both. 8 GB inference does not imply 8 GB training.
- Institutional permission for use of the UC-PnC manuals, and ethics clearance if student queries are collected. Longest lead time of anything here — start it first.
