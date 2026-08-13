# Objective 5 — Query-Routing Mechanism: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two-gate, three-outcome query router (cited answer / office referral / out-of-scope) as a standalone, calibrated, testable proof-of-concept, and prove on the 80-query labeled set that it separates the three categories before it is wired into the rest of the pipeline.

**Architecture:** Gate 2 (evidence sufficiency) runs first over Objective 4's hybrid retrieval output, using a disjunctive component gate (`lexical_ok OR dense_ok`). Only when Gate 2 says *insufficient* does Gate 1 (scope) run, deciding the *reason* for the non-answer via a frozen labeled nearest-neighbour lookup against exemplar queries. The referral path assembles the top 2–3 candidate `issuing_office` values from the retrieval ranking and hands only those short names to the SLM under a narrow single-purpose prompt, with a deterministic override when the model names an off-list office. Out-of-scope is a static string with zero model calls.

**Tech Stack:** Python 3.13 (WSL venv at `venv/`), `rank_bm25`, `sentence-transformers` (`bge-small-en-v1.5`, CPU), `numpy`, `pytest` (new), `openai` client as the Gate-3 placeholder until the local 4-bit SLM lands.

---

## Global Constraints

Copied verbatim or near-verbatim from the Notion ground truth and repo governing docs. Every task's requirements implicitly include this section.

- **Objective 5 only.** Objectives 1, 2, 3, 4, and 6 are not modified by this plan. The one exception is a single additive change to `rag_pipeline/build_corpus.py` (Objective 3's build step) to stamp `issuing_office` — this is Objective 3's already-decided deliverable, consumed here.
- **Objective 4's hybrid retrieval (BM25 + dense embedding + RRF + disjunctive sufficiency gate) is fixed, already-decided architecture.** Reuse its outputs. Do not redesign it, do not re-evaluate it, do not run a retrieval ablation.
- **Objective 3's office lookup table is an existing dependency.** Build on the `issuing_office` field. Do not redesign how it is derived.
- **Metrics are frozen at exactly three:** routing accuracy, coverage, selective risk. No new named metric. Office-selection correctness is reported as a *breakdown within* routing accuracy, never as a fourth metric.
- **No new agents. No additional classifiers beyond the exemplar-based scope check. No vector database, no ANN index (Chroma/FAISS/Pinecone/Qdrant/pgvector).** Gate 1 stays a frozen nearest-neighbour lookup — no training, no gradient update.
- **Neither gate is an LLM call.** Both are numeric/similarity comparisons. The routing decision never depends on the quantized model's own confidence.
- **The query is embedded exactly once per query** (`bge-small-en-v1.5`) and the vector is reused by Gate 1 and Gate 2's dense half.
- **`EMBEDDING_DEVICE = "cpu"`, load-bearing.** The 4-bit SLM needs all 8 GB of VRAM. Gate 1 and dense retrieval must allocate zero VRAM.
- **All tunables live in `routing/config.py`.** Never hardcode a threshold inside a gate.
- **Two model calls maximum per query, and only one at a time:** 1 for the answer path (Gate 3), 1 short call for the referral path, 0 for out-of-scope. `generator_calls` remains a cheap correctness check on the routing table.
- **English only.** No clarification route. No reranking beyond RRF. No multilingual support.
- **`document_id` must stay fixed-width and prefix-safe.** Current scheme is already compliant: `HDBK-0079`, `INTN-0001`, `RSCH-0042`.
- **Freeze `chunk_documents.py` and `CHUNK_SIZE_CAP` before the eval set is labeled.** Gold labels reference chunk ids; re-chunking silently invalidates them.
- **Governing authority:** `THESIS_BATTLE_PLAN.md` overrides `README.md`, `ArchitectureAndDesign.md`, `prompt_and_routing_architecture.md`, `EVALUATION_PLAN.md`. `consultation/*` is superseded — design rationale only, never cited, scope never reintroduced.
- **Commit after every task.** Repo is a git repo on `main` with 11 files already modified — commit or stash that work before starting Task 1.

---

## What I found in Notion, and what the corpus actually contains

Read as ground truth (in precedence order): **Objectives v2** (`Thesis Objectives — Planning Draft v2`, the technical objectives), **Objectives v3** (`Thesis Objectives — Submission Wording (short form)`, the simplified submission sentences), `Offline University Helpdesk Prototype (Thesis)`, `Prompt and Routing Architecture`. Objectives **v1 was not used** (archive, per instruction). Repo-side: `CLAUDE.md`, `THESIS_BATTLE_PLAN.md`, `EVALUATION_PLAN.md`, `docs/superpowers/specs/2026-08-04-hybrid-retrieval-and-calibration-design.md`, and the live `routing/` code and data.

**Confirmed from Notion — the design in the task description matches ground truth exactly:**

- Gate 2 before Gate 1 is a *deliberately resolved* decision, not an accident. Gate 1 only decides the *reason* for a non-answer. `scope_similarity` is logged on every query including the answer path, so the false-rejection rate under the old ordering is still reportable.
- Three outcomes, three metrics (routing accuracy / coverage / selective risk), 80-query set stratified ~40 / ~25 / ~15.
- The referral path is model-assisted (v2 changelog item 4 explicitly removed the word "predefined"), bounded to top 2–3 candidate office names under a narrow prompt, with a deterministic override on off-list output.
- Out-of-scope is a fully static string, zero cost.
- Gate 1 stays an exemplar-based nearest-neighbour lookup. "Calibrated classifier research" is in the *Excluded* list.
- v2's Data Sourcing Appendix already recorded that `issuing_office` does not exist in the corpus and that a versioned section→office lookup table is the approved fix.

**Verified directly against `routing/data/corpus.json` (364 passages) — five findings that change the plan:**

1. **`issuing_office` genuinely does not exist.** Fields are exactly `document_id, title, source, revision_date, page_or_section, text`. Confirms v2's appendix. Task 1 adds it.
2. **The office lookup table is ~2× larger than v2 estimated.** v2 says "~25 rows". Actual distinct `(document_prefix, section_root, chapter)` keys: **73** — of which **19 are front-matter pages with no breadcrumb at all** (`page_or_section` is bare `"p. 4"`), leaving **54 substantive section keys**. Still one-time bounded work, but budget ~54 rows plus 3 document-level defaults, not 25.
3. **`revision_date` is `"not available"` on all 364 passages (100%).** Matches v2's Finding 2. Objective 5 does not depend on this field, so it is **not** a blocker here — flagged only because it is Objective 3's open item.
4. **`scope_exemplars.json` is unusable as-is for Gate 1** — this is the largest gap between the design and the code. It holds 20 in-scope exemplars, **no out-of-scope exemplars** (v2 says "in-scope/out-of-scope"), and the content is generic US-university phrasing ("financial aid", "add/drop period", "dispute a parking ticket", "registrar's office") that does not describe the UC-PnC corpus. Task 4 replaces it.
5. **Gate 2 is BM25-only today and its threshold is a measured no-op.** `BM25_COVERAGE_THRESHOLD = 1.0` admits everything; the hybrid spec measured in-scope top-1 at 9.80–23.07 against out-of-scope junk at 4.65–11.02 — **overlapping**, so no absolute threshold separates them under the current `text.lower().split()` tokenizer. The referral route currently never fires on retrieval grounds. This is Objective 4's item to fix (its work-order step 4) and is a **hard external prerequisite** for Objective 5's calibration — see "External prerequisites" below.

**Stale-doc note (not Objective 5's business, flagged so it isn't inherited):** the repo's `EVALUATION_PLAN.md` and `CLAUDE.md` still describe Configuration C (8-bit reference) and an unnamed SLM. Notion v2 removed Configuration C entirely and v3 names Llama 3.2 3B. v2's own sync backlog already tracks this. This plan follows **Notion**, and touches none of those files.

---

## External prerequisites (Objective 4 / Objective 3 — not built here)

Objective 5 consumes these. It can be **built and unit-tested today** against a fake retriever, but it **cannot be calibrated or validated on the target architecture** until they land.

| # | Prerequisite | Owner | Blocks |
|---|---|---|---|
| P1 | Tokenizer fix in `retriever.py` (regex + stopwords + Snowball stemming, identical at index and query time) | Objective 4, its work-order step 4 | **All** threshold calibration. Without it the measured distributions overlap and no operating point exists. |
| P2 | `DenseRetriever` + `embeddings.npy` + `manifest.json`, with the `bge` query-side instruction prefix applied to the query only | Objective 4, steps 5–6 | `dense_ok` half of the disjunctive gate; the "only dense fires" path |
| P3 | RRF fusion with `RRF_K` exposed as a calibratable tunable | Objective 4, step 6 | Office-candidate ranking on the refer path |
| P4 | `office_lookup.json` applied at corpus-build time | Objective 3 | The entire refer path. **Task 1 of this plan builds it**, since v2 lists it as an unblocked internal engineering artifact with no adviser dependency. |

**Mitigation — two-pass validation, and it is a feature not a workaround:**

- **Pass 0 (runnable as soon as P1 lands):** run the full harness with `dense_ok` hardwired `False`. Proves the harness, the eval data, the exemplar set, and the referral path end-to-end, and produces a BM25-only separation baseline.
- **Pass 1 (after P2–P3):** rerun the *identical* sweep with the real hybrid `Gate2Signals`. **Only Pass 1's numbers appear in the thesis.** Pass 0 is a build-correctness check, not a result — do not report it as one.

**The contract between Objective 4 and Objective 5 is exactly two methods plus one dataclass.** Objective 4's retriever must expose:

```python
def signals_for(self, query: str, query_embedding) -> Gate2Signals: ...
def passages_for(self, signals: Gate2Signals) -> List[Passage]: ...
```

`Gate2Signals` (Task 2) carries `coverage`, `norm_bm25`, `dense_max_cos`, `bm25_ranking`, `dense_ranking`, where the rankings are `List[Tuple[passage_index, score]]` indexing into `passages_for(...)`. Objective 4 fills these; Objective 5 only reads them. Nothing in this plan reaches inside the retriever, so Objective 4 stays free to change how BM25, dense, and RRF are implemented behind that surface.

---

# SECTION 1 — ARCHITECTURE

## 1.1 The two-gate flow

```text
English query
  │
  ├─ embed ONCE: bge-small-en-v1.5, CPU, query-side instruction prefix
  │    ("Represent this sentence for searching relevant passages: ")
  │    → one 384-d L2-normalized vector, reused by Gate 2 dense AND Gate 1
  │
  ▼
GATE 2 — EVIDENCE SUFFICIENCY  [Objective 4's retrieval, read by Objective 5]
  BM25 top-N   ──► lexical_ok = (coverage ≥ c) OR (norm_bm25 ≥ s)
  dense top-N  ──► dense_ok   = (max_cosine ≥ d)
  sufficient   = lexical_ok OR dense_ok          ← DISJUNCTIVE
  │
  ├── sufficient = True ─────────────────────────────────────► PATH A
  │     RRF fuses ONLY the lists whose own gate fired
  │     top_k passages → GATE 3 (generator, 1 model call)
  │     decision = "answer"          generator_calls = 1
  │
  └── sufficient = False
        │
        ▼
      GATE 1 — SCOPE CHECK  [reuses the query vector; runs, but is only ACTED ON here]
        1-NN over the labeled exemplar union (in_scope + out_of_scope)
        in_scope = (nearest exemplar's label == "in_scope")
                   AND (max_similarity ≥ SCOPE_SIMILARITY_THRESHOLD)
        │
        ├── in_scope = False ──────────────────────────────────► PATH C
        │     decision = "out_of_scope"
        │     answer   = config.SCOPE_MESSAGE   (static string)
        │     generator_calls = 0               (zero cost)
        │
        └── in_scope = True ───────────────────────────────────► PATH B
              office candidates = top 2–3 DISTINCT issuing_office values,
                ranked by referral-only RRF over BOTH candidate lists
              │
              ├── 0 candidates  → generic REFERRAL_MESSAGE, generator_calls = 0
              ├── 1 candidate   → that office, templated, generator_calls = 0
              └── 2–3 candidates→ narrow referral prompt, 1 short model call
                                   + deterministic off-list override
              decision = "refer"
```

## 1.2 The three paths, precisely

| Path | Trigger (exact condition) | What happens | `generator_calls` |
|---|---|---|---|
| **A — cited answer** | `gate2.sufficient == True` | RRF over the passed lists → `top_k` passages → Gate 3 with `SYSTEM_PROMPT` → citation validation against supplied `document_id`s | 1 |
| **B — office referral** | `gate2.sufficient == False` **AND** `gate1.in_scope == True` | Assemble 2–3 distinct candidate offices → narrow referral prompt → validate/override → templated or validated-model sentence | 0 or 1 |
| **C — out-of-scope** | `gate2.sufficient == False` **AND** `gate1.in_scope == False` | Return `config.SCOPE_MESSAGE` verbatim | 0 |

**The invariant that makes the ordering worth defending:** there is no path from a Gate-2-sufficient query to `out_of_scope`. Gate 1's decision is computed on every query (so `scope_similarity` is always logged) but is structurally unreachable on Path A. `test_router_paths.py` asserts this directly — an answerable query with an adversarially low scope similarity must still route to `answer`. That test *is* the false-out-of-scope failure mode, encoded.

## 1.3 One decision this plan makes that the notes did not specify

**Referral-only fusion (D-1).** On Path B, *neither* retriever's gate fired, so §2.3 of the hybrid spec says fuse nothing. But Path B still needs a ranking to pull the top 2–3 offices from. Decision: run RRF over both full candidate lists **unconditionally, for office ranking only** — the fused list is never sent to a generator as evidence and never influences `sufficient`.

This does not violate the hybrid spec's separation of concerns ("fusion decides order, the gate decides whether to answer"): the gate has already decided *not* to answer. Fusion is only choosing which office name to put in front of the student. Cost is zero — both lists are already computed. The alternative (referring with no office named) throws away the entire point of Objective 3's `issuing_office` field.

---

# SECTION 2 — DESIGN

## 2.1 How to set the Gate 2 thresholds

Three thresholds, all in `config.py`, all calibrated together — never set by eye.

| Name | Symbol | Signal | Grid |
|---|---|---|---|
| `BM25_COVERAGE_THRESHOLD` | `c` | fraction of the query's post-stemming content terms present in the BM25 top-1 passage | `0.0 → 0.8` step `0.1` (9) |
| `BM25_SCORE_THRESHOLD` | `s` | **length-normalized** BM25 top-1 (see below) | `0.0 → 0.50` step `0.05` (11) |
| `DENSE_SCORE_THRESHOLD` | `d` | max cosine over the dense top-N | `0.30 → 0.80` step `0.05` (11) |

**Why the BM25 score must be normalized, not raw.** Raw BM25 does not transfer across query lengths — a 3-word query and a 15-word query live on different scales, so one absolute cut-off cannot serve both. Normalize by the query's own saturation ceiling:

```python
norm_bm25 = top1_raw_score / max(1e-9, (k1 + 1.0) * sum(idf(t) for t in query_terms if t in vocab))
```

`(k1+1) * Σ idf` is the score the passage would earn if every query term matched with unbounded term frequency, so `norm_bm25 ∈ [0, 1]`. Terms absent from the vocabulary contribute nothing to either numerator or denominator, so an all-OOV query yields `0.0` rather than a divide-by-zero. `BM25Okapi` exposes `k1` and `idf` directly.

**Selection rule — pre-register this before running the sweep.** Do not maximize a blended score; state a constraint and satisfy it:

> Among all `(c, s, d)` on the grid, choose the point that **maximizes Gate-2 recall on the ~40 answerable queries, subject to a false-accept rate ≤ 0.10 on the ~40 non-answerable queries** (the 25 in-scope-no-evidence and 15 out-of-scope strata combined). Break ties toward the higher `d` (dense is the paraphrase safety net) and then toward the lower `c`.

Publish the whole sweep surface, not just the winner. The chosen point is then a *disclosed engineering decision*, which is defensible, rather than a fitted parameter presented as a finding, which is not.

**The overfitting problem, and its in-scope fix.** Thresholds fitted on the same 80 queries used to report routing accuracy inflate that accuracy. Fix: **5-fold grouped cross-validation**, grouped by `paraphrase_family` so no family straddles a fold boundary. For each fold, select thresholds on the other four and evaluate on the held-out one; report mean ± spread as the headline routing accuracy, alongside the single frozen operating point selected on all 80 (which is what ships in `config.py`). This is not a new metric — it is a less-wrong estimator of routing accuracy — and it costs nothing, because Gate 1 and Gate 2 are deterministic arithmetic over cached per-query signals. No model call is made anywhere in the sweep.

## 2.2 How the Gate 1 exemplar set is built and sized

**Decision rule (D-2): labeled 1-NN with a distance floor.**

```python
in_scope = (label_of_argmax_cosine == "in_scope") and (max_cosine >= SCOPE_SIMILARITY_THRESHOLD)
```

The label test does the work; the threshold guards the case where a query is far from *everything* but happens to lean marginally in-scope. Both are swept together (`SCOPE_SIMILARITY_THRESHOLD` over `0.30 → 0.85` step `0.05`, 12 points). This is still one frozen lookup with no training — it is exactly v2's "nearest-neighbour distance to a fixed set of exemplar in-scope/out-of-scope queries", and it is strictly simpler than the two-threshold alternative.

**The sizing insight that matters most.** ~25 of the 80 eval queries are *in scope but have no corpus evidence* — Gate 1 must call these **in-scope**. If the exemplars are drawn only from corpus topics, every one of those 25 lands nearest to nothing and routes to `out_of_scope`, and the referral path — the entire novel contribution of Objective 5 — silently dies. **The in-scope exemplar set therefore needs two families, and the second one is the load-bearing one:**

| Family | Label | Count | Content | Why |
|---|---|---|---|---|
| `corpus_topic` | `in_scope` | ~30 | One per substantive chapter key a student could plausibly ask about, spread across all three manuals, in student register | Path A and the answerable stratum |
| `university_not_in_corpus` | `in_scope` | ~15 | Real UC-PnC student concerns the three manuals do **not** cover — dormitory/housing, ID replacement fee, shuttle service, canteen, tuition instalment schedules, campus WiFi | **Makes Path B reachable.** Without this family Gate 1 cannot distinguish "in scope, no evidence" from "not our problem" |
| `out_of_scope` | `out_of_scope` | ~20 | 4 each across 5 named categories: general knowledge, other institutions, personal chit-chat, coding/homework help, local services & commerce | Path C |

**Total ~65 exemplars.** Size by *coverage*, not by count: every substantive chapter key should have at least one in-scope neighbour, and every out-of-scope category at least four. Adding exemplars past coverage buys nothing in a 1-NN rule.

**Leakage rule, enforced by test, not by discipline:** no exemplar may be an eval query verbatim or a near-paraphrase. `test_eval_data.py` computes the full exemplar × eval-query cosine matrix and fails if any pair exceeds `0.95`, listing every violation. Author exemplars *before* or independently of the eval set; if a collision surfaces, rewrite the **exemplar**, never the eval query.

## 2.3 The office-referral prompt

**Candidate assembly (deterministic, before any prompt):** take the referral-only RRF ranking (D-1), map each passage to its `issuing_office`, drop `"Unknown"`, deduplicate **preserving rank order**, keep the first `REFERRAL_MAX_CANDIDATES = 3`.

Short-circuits that cut model calls without losing anything:

- **0 candidates** → generic `config.REFERRAL_MESSAGE`, no office named, `generator_calls = 0`
- **1 candidate** → template that office directly, `generator_calls = 0` (nothing to disambiguate)
- **2–3 candidates** → the one narrow model call below

**System prompt** (`config.REFERRAL_SYSTEM_PROMPT`), single-purpose and deliberately claustrophobic:

```text
You route university helpdesk questions to the correct office.
You are given a student's question and a numbered list of candidate offices.
Choose exactly one office from the list and write one short sentence telling the student to contact it.
Never name an office that is not in the list. Never answer the question itself.
Never state contact details, room numbers, hours, fees, deadlines, or web addresses.
Reply with the office name on the first line and the sentence on the second line. Nothing else.
```

**User message:**

```text
Question: {query}

Candidate offices:
1. {office_1}
2. {office_2}
3. {office_3}
```

Decoding: `temperature = 0.0`, `max_tokens = 80`. No passages, no evidence, no corpus text — just a question and 2–3 short strings, so the prefill cost is trivial and there is nothing to hallucinate *from*.

**Deterministic override (`REFERRAL_OVERRIDE`), applied to the selection:**

1. Normalize (casefold, collapse whitespace) the model's first line and every candidate.
2. Exactly one candidate matches → accept it.
3. Zero matches, or two or more matches, or a leading index outside `1..len(candidates)` → **override to `candidates[0]`** (the top-ranked office) and set `referral_override = True`.

`referral_override` is logged on every referral. A high override rate is direct evidence the prompt is too loose — report it as a prompt-quality signal, not as a metric.

**Phrasing (`REFERRAL_USE_MODEL_PHRASING`, default `True`).** v2 asks the model to phrase the sentence, so keep the model's second line — but only when it survives validation: it must contain the *selected* office name, must contain no digit, no `@`, and no `http`, and must be ≤ 200 characters. Any failure falls back to the template:

```python
REFERRAL_TEMPLATE = (
    "I could not verify this from the selected LMS materials. "
    "Please contact the {office} for this concern."
)
```

The digit/`@`/`http` filter is the point: those are exactly the characters an invented phone number, email, room number, or URL would arrive in. Setting the flag `False` makes the surface 100% templated — worth reporting as a fallback if validation failures turn out to be common.

## 2.4 Where every decision point lives in code

| Decision | Module | Function / constant |
|---|---|---|
| `lexical_ok`, `dense_ok`, `sufficient` | `routing/gate2.py` | `Gate2Signals.evaluate()` |
| `norm_bm25` computation | `routing/gate2.py` | `normalized_bm25()` |
| coverage computation | `routing/gate2.py` | `term_coverage()` |
| `c`, `s`, `d` values | `routing/config.py` | `BM25_COVERAGE_THRESHOLD`, `BM25_SCORE_THRESHOLD`, `DENSE_SCORE_THRESHOLD` |
| labeled 1-NN scope rule | `routing/scope_gate.py` | `ScopeGate.check()` |
| `SCOPE_SIMILARITY_THRESHOLD` | `routing/config.py` | — |
| referral-only RRF (D-1) | `routing/referral.py` | `rank_office_candidates()` |
| candidate dedupe + cap | `routing/referral.py` | `rank_office_candidates()` |
| referral prompt text | `routing/config.py` | `REFERRAL_SYSTEM_PROMPT`, `REFERRAL_TEMPLATE` |
| off-list override | `routing/referral.py` | `select_office()` |
| model-phrasing validation | `routing/referral.py` | `validate_phrasing()` |
| three-path branching | `routing/router.py` | `Router.route()` |
| section → office mapping | `rag_pipeline/data/office_lookup.json` | data, versioned |
| lookup application | `rag_pipeline/build_corpus.py` | `resolve_issuing_office()` |
| threshold sweep + fold CV | `routing/eval/calibrate.py` | `sweep()`, `grouped_cv()` |
| the three metrics | `routing/eval/report.py` | `routing_accuracy()`, `coverage()`, `selective_risk()` |

---

# SECTION 3 — DATA

## 3.1 `routing/data/corpus.json` — schema v2

Existing fields are unchanged. **One field is added.** Everything else in this file stays exactly as Objective 3 produces it.

```json
{
  "passages": [
    {
      "document_id": "HDBK-0079",
      "title": "UC-PnC Student Handbook Manual",
      "source": "University LMS / UC-PnC Student Handbook Manual",
      "revision_date": "not available",
      "page_or_section": "PART 2. ACADEMIC GUIDELINES > CHAPTER V SENIOR HIGH SCHOOL, COLLEGIATE, AND GRADUATE SCHOOL GRADING SYSTEM > Section 2. Specific Policies (p. 92)",
      "issuing_office": "Office of the University Registrar",
      "text": "..."
    }
  ]
}
```

`Passage` gains `issuing_office: str = "Unknown"` **with a default**, so `Passage(**doc)` in `retriever.py` keeps loading a pre-migration `corpus.json` instead of raising. Objective 5 must not be the reason an old artifact stops loading.

`page_or_section` is the *only* structural handle available: the corpus carries no separate `section_path` field (that lives upstream in `rag_pipeline/data/chunks/*.jsonl`). Parse the breadcrumb out of it by stripping the trailing `(p. N)` / `(pp. N-M)` and splitting on `>`.

## 3.2 `rag_pipeline/data/office_lookup.json` — the Objective 3 dependency

Versioned, hand-authored, auditable — a researcher-constructed artifact, cited as such in Chapter 3.

```json
{
  "version": 1,
  "created": "2026-08-08",
  "note": "Section->office mapping. Hand-built from the three UC-PnC manuals' own organizational structure. Researcher-constructed; NOT extracted from the source text.",
  "office_vocabulary": [
    "Office of the University Registrar",
    "Office of Student Affairs and Services",
    "Guidance and Counseling Office",
    "University Library",
    "Research and Development Office",
    "Knowledge Management and Innovation Department",
    "Placement, Alumni and Linkages Office",
    "Medical and Dental Clinic",
    "Sports Development Office",
    "Scholarship Office",
    "Office of Admissions",
    "Information Technology Office",
    "Office of the Vice President for Academic Affairs"
  ],
  "document_defaults": {
    "HDBK": "Office of Student Affairs and Services",
    "INTN": "Placement, Alumni and Linkages Office",
    "RSCH": "Research and Development Office"
  },
  "rules": [
    { "doc": "HDBK", "root": "PART 3. STUDENT SUCCESS", "chapter": "CHAPTER IX LIBRARY SERVICES",            "office": "University Library" },
    { "doc": "HDBK", "root": "PART 3. STUDENT SUCCESS", "chapter": "CHAPTER VII GUIDANCE AND COUNSELING PROGRAM AND SERVICES", "office": "Guidance and Counseling Office" },
    { "doc": "HDBK", "root": "PART 2. ACADEMIC GUIDELINES", "chapter": "CHAPTER II ISSUANCE OF STUDENTS ACADEMIC RECORDS", "office": "Office of the University Registrar" },
    { "doc": "RSCH", "root": "GUIDELINES", "chapter": "CHAPTER V INTELLECTUAL PROPERTY RIGHTS",              "office": "Knowledge Management and Innovation Department" }
  ]
}
```

**Sizing, measured not estimated:** 73 distinct `(doc, root, chapter)` keys exist. 19 are bare front-matter pages (`page_or_section == "p. 4"`, no breadcrumb) and are served by `document_defaults`. **54 substantive rules to author.** At ~30 seconds each that is under half an hour — still comfortably "one-time, bounded", just twice v2's ~25-row estimate.

**Resolution order** (`resolve_issuing_office`), most specific first:

1. exact `(doc, root, chapter)`
2. `(doc, root)` — any chapter under that part
3. `document_defaults[doc]`
4. `"Unknown"` — must never occur; `test_corpus_schema` fails the build if any passage resolves to it

The office vocabulary is closed. `expected_office` in the eval set must draw from `office_vocabulary`, validated by test — otherwise routing accuracy's office breakdown compares strings that can never match.

## 3.3 `routing/data/eval_queries.json` — the 80-query set

Authored by the user (domain-authored labels are what make the writeup defensible; see `EVALUATION_PLAN.md`). Source from real students where possible and disclose it as a limitation if not.

```json
{
  "version": 1,
  "corpus_hash": "sha256:...",
  "queries": [
    {
      "query_id": "Q001",
      "query": "How is the final grade computed for a college subject?",
      "stratum": "answerable",
      "expected_decision": "answer",
      "expected_office": null,
      "gold_document_ids": ["HDBK-0079", "HDBK-0080"],
      "source_document": "UC-PnC Student Handbook Manual",
      "paraphrase_family": "grading_computation",
      "notes": ""
    },
    {
      "query_id": "Q055",
      "query": "How much does it cost to replace a lost student ID?",
      "stratum": "in_scope_no_evidence",
      "expected_decision": "refer",
      "expected_office": "Office of Student Affairs and Services",
      "gold_document_ids": [],
      "source_document": null,
      "paraphrase_family": "id_replacement",
      "notes": "Manuals mention the ID but not the replacement fee."
    },
    {
      "query_id": "Q071",
      "query": "What is the best way to cook chicken adobo?",
      "stratum": "out_of_scope",
      "expected_decision": "out_of_scope",
      "expected_office": null,
      "gold_document_ids": [],
      "source_document": null,
      "paraphrase_family": "cooking",
      "notes": ""
    }
  ]
}
```

| Stratum | Count | `expected_decision` | `expected_office` | `gold_document_ids` |
|---|---|---|---|---|
| `answerable` | ~40 | `answer` | `null` | **required, non-empty** |
| `in_scope_no_evidence` | ~25 | `refer` | **required, ∈ `office_vocabulary`** | `[]` |
| `out_of_scope` | ~15 | `out_of_scope` | `null` | `[]` |

**Authoring constraints, all machine-checked by `test_eval_data.py`:**

- `query_id` unique; `stratum` → `expected_decision` mapping exact per the table.
- `paraphrase_family` present on every row. **It is the CV grouping key** — two paraphrases of the same underlying question must never sit in different folds, or the CV estimate leaks. Families should hold 1–3 queries.
- Stratify by *source document*, never randomly — all three manuals represented in the answerable stratum roughly in proportion to their chunk counts (234 / 108 / 22).
- No eval query may resemble a QLoRA training question (`EVALUATION_PLAN.md`) or a Gate 1 exemplar (§2.2).
- `gold_document_ids` must all exist in `corpus.json`.
- `corpus_hash` pins the corpus the labels were authored against. A rebuild that changes the hash **flags the eval set stale** — the harness warns loudly and does not silently proceed.

## 3.4 `routing/data/scope_exemplars.json` — v2 shape

Replaces the current 20 generic in-scope strings.

```json
{
  "version": 2,
  "exemplars": [
    { "query": "How do I borrow a book from the university library?",
      "label": "in_scope", "family": "corpus_topic", "topic": "library" },
    { "query": "How much is the fee to replace a lost student ID?",
      "label": "in_scope", "family": "university_not_in_corpus", "topic": "student_services" },
    { "query": "What is the capital of Japan?",
      "label": "out_of_scope", "family": "out_of_scope", "topic": "general_knowledge" }
  ]
}
```

`ScopeGate` must load **both** shapes: the v2 object above, and the legacy `{"in_scope_examples": [str, ...]}` (normalized to `label="in_scope"`). Five lines, and it keeps the demo working mid-migration.

---

# SECTION 4 — PoC VALIDATION PLAN

Prove the router separates the three categories on real data **before** wiring it into the pipeline. The entire validation below — except §4.4 — makes **zero model calls**: Gate 1 and Gate 2 are deterministic arithmetic, so the sweep runs on cached per-query signals in seconds.

## 4.1 What to run

```bash
# from routing/, via the WSL venv (see CLAUDE.md)
../venv/bin/python -m eval.run_poc --stage signals    # 80 queries -> cached gate signals (no LLM)
../venv/bin/python -m eval.run_poc --stage sweep      # threshold grid + 5-fold grouped CV
../venv/bin/python -m eval.run_poc --stage report     # confusion matrix + the three metrics
../venv/bin/python -m eval.run_poc --stage referral   # ONLY stage that calls the model (~25 queries)
```

`--stage signals` writes `routing/eval/out/signals.json`: one row per query with `norm_bm25`, `coverage`, `dense_max_cos`, `scope_max_sim`, `scope_nn_label`, `office_candidates`, plus the corpus hash. Everything downstream reads that file, so the sweep never re-runs retrieval.

## 4.2 Pre-registered success criteria — write these down before looking at results

| # | Check | Passes if | Fails if |
|---|---|---|---|
| **V1** | **Gate 2 separation.** Distribution of `max(norm_bm25, dense_max_cos)` per stratum | An operating point exists with ≥ 0.90 recall on `answerable` at ≤ 0.10 false-accept on the other 40 | The distributions overlap the way the hybrid spec measured for BM25-only. **No threshold works** — do not keep sliding thresholds; the sufficiency *signal* is wrong |
| **V2** | **Gate 1 separation**, measured *only* on queries Gate 2 rejected | ≥ 0.80 of `in_scope_no_evidence` → in-scope **and** ≥ 0.80 of `out_of_scope` → out-of-scope | Either side below 0.80. If `in_scope_no_evidence` is the weak side, the `university_not_in_corpus` exemplar family is too thin — **fix the exemplars, not the threshold** |
| **V3** | **Routing accuracy**, full 3×3 confusion matrix at the frozen point | ≥ 0.80 overall | Below 0.80, or heavy mass in any single off-diagonal cell |
| **V4** | **The ordering invariant.** Count of `answerable → out_of_scope` | **Exactly 0** | Any nonzero count is a *code* defect, not a metric dip — Gate 1 is being consulted on Path A. Stop and fix `router.py` |
| **V5** | **Coverage** = `answer`-routed / 80 | Within `0.35 – 0.60` (ground truth is 40/80 = 0.50) | `< 0.30` over-refusing; `> 0.70` over-answering. Both make the metric pair meaningless |
| **V6** | **Selective risk** = fraction of `answer`-routed queries whose true stratum ≠ `answerable` | ≤ 0.15 | Above 0.15 with high coverage = the disjunctive gate is too permissive; report it rather than quietly tightening past the pre-registered rule |
| **V7** | **Cross-validated routing accuracy**, 5-fold grouped by `paraphrase_family` | Within ~0.05 of the in-sample number | A large gap means the thresholds are fitted to 80 points. **Report the CV number as the headline**, not the in-sample one |
| **V8** | **Office accuracy within the refer path** (a breakdown of V3, not a fourth metric) | Model selection ≥ deterministic top-1 | Model ≤ deterministic top-1 → **drop the model call.** A referral path that is pure `candidates[0]` is cheaper, fully deterministic, and one less thing to defend |
| **V9** | **Override rate** = referrals where the model named an off-list office | Near 0 | High → the prompt is too loose. Prompt-quality signal, not a metric |

## 4.3 What "it doesn't work" looks like, and the honest response to each

- **V1 fails.** The measured overlap from the hybrid spec (out-of-scope junk at 11.02 outscoring an in-scope query at 9.80) persisted through the tokenizer fix. The referral route cannot fire on retrieval grounds. Correct response: report it, and change the *signal* — not slide the threshold until the confusion matrix looks nicer.
- **V2 fails on the in-scope side.** Gate 1 is collapsing `refer` into `out_of_scope`. Almost always the `university_not_in_corpus` exemplar family is too thin. Add exemplars, re-run; do not lower the threshold, which would drag genuine out-of-scope queries in with them.
- **V4 fails.** Structural bug. Gate 1 is reachable on the answer path — the exact failure mode the Gate 2 → Gate 1 ordering was resolved to eliminate.
- **V5 and V6 disagree** (high coverage, high selective risk). The disjunctive gate is admitting weak matches — a cost the hybrid spec accepted deliberately (§2.2). Report the trade-off, backed by the prompt contract and citation validation, rather than switching to a conjunctive gate, which would re-introduce exactly the paraphrase failure the disjunction exists to prevent.
- **V8 fails.** Good news, not bad: delete the referral model call. Fewer moving parts, zero generation cost on Path B, and a scope reduction the adviser will welcome.

## 4.4 Scope note on selective risk at PoC stage

Objective 5's PoC measures **routing-level** selective risk: of the queries routed to `answer`, how many were not actually answerable. It cannot measure whether the generated answers were *correct* — that requires the human correctness/groundedness grading owned by Objectives 2 and 4.

So the PoC number is a **lower bound** on the reported selective risk. Say so in Chapter 3 rather than letting the two definitions blur. This is a reporting note, not a new metric.

---

# SECTION 5 — FILE / MODULE LAYOUT

```text
rag_pipeline/
  data/office_lookup.json          [NEW]  versioned section->office table (~54 rules + 3 defaults)
  build_corpus.py                  [MOD]  + resolve_issuing_office(), stamps every passage

routing/
  config.py                        [MOD]  + BM25_SCORE_THRESHOLD, REFERRAL_* , sweep grids
  retriever.py                     [MOD]  Passage.issuing_office (defaulted). Retrieval logic = Objective 4's
  gate2.py                         [NEW]  Gate2Signals: coverage / norm_bm25 / dense_max_cos -> sufficient
  scope_gate.py                    [MOD]  labeled 1-NN rule + v2 exemplar loader
  referral.py                      [NEW]  office ranking, narrow prompt, override, phrasing validation
  router.py                        [MOD]  three-path wiring + new RouteResult fields
  generator.py                     [UNCHANGED — Gate 3 is Objective 4's]
  main.py                          [MOD]  print the new referral fields in the demo

  data/
    corpus.json                    [REBUILT]  + issuing_office
    scope_exemplars.json           [REPLACED] v2 labeled shape, ~65 exemplars
    eval_queries.json              [NEW]      80 labeled queries

  eval/
    __init__.py                    [NEW]
    harness.py                     [NEW]  run 80 queries -> cached per-query gate signals (no LLM)
    calibrate.py                   [NEW]  threshold sweep + 5-fold grouped CV + operating point
    report.py                      [NEW]  3x3 confusion, routing accuracy, coverage, selective risk
    run_poc.py                     [NEW]  CLI: --stage signals|sweep|report|referral
    out/                           [gitignored] signals.json, sweep.csv, report.md

  tests/
    conftest.py                    [NEW]  puts routing/ on sys.path (existing code uses flat imports)
    test_gate2_sufficiency.py      [NEW]
    test_scope_gate.py             [NEW]
    test_referral.py               [NEW]
    test_router_paths.py           [NEW]  includes the V4 ordering invariant
    test_eval_data.py              [NEW]  schema + leakage + office-vocabulary validation
```

**Boundaries.** `gate2.py` owns *reading* Objective 4's retrieval output and turning it into a sufficiency decision — it never retrieves. `referral.py` owns everything on Path B and knows nothing about Gate 3. `router.py` owns only the branching and stays readable end to end. `eval/` is a leaf: it imports the gates, nothing imports it.

**Conventions to match.** Existing modules use flat imports (`import config`) and are run from inside `routing/`. Keep that — `conftest.py` handles it for tests rather than restructuring working code. Dependencies live in the WSL venv at `venv/`; `pytest` must be added to `routing/requirements.txt`.

---

# SECTION 6 — TASKS

### Task 1: Office lookup table and `issuing_office` in the corpus

**Files:**
- Create: `rag_pipeline/data/office_lookup.json`
- Modify: `rag_pipeline/build_corpus.py`
- Modify: `routing/retriever.py` (the `Passage` dataclass only)
- Test: `routing/tests/test_corpus_schema.py`, `routing/tests/conftest.py`

**Interfaces:**
- Consumes: `routing/data/corpus.json` (6-field shape), `rag_pipeline/data/chunks/*.jsonl`
- Produces: `resolve_issuing_office(document_id: str, page_or_section: str, table: dict) -> str`; `corpus.json` with `issuing_office` on all 364 passages; `Passage.issuing_office: str = "Unknown"`

- [ ] **Step 1: Add pytest and the test path shim**

Append to `routing/requirements.txt`:
```
pytest>=8.0.0
```

Create `routing/tests/conftest.py`:
```python
import os
import sys

# Existing routing modules use flat imports (`import config`) and are run from
# inside routing/. Mirror that for tests instead of restructuring working code.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 2: Write the failing test**

Create `routing/tests/test_corpus_schema.py`:
```python
import json

import config
from retriever import Passage
from rag_pipeline_office import resolve_issuing_office, load_office_table


def test_resolve_prefers_exact_chapter_over_document_default():
    table = {
        "document_defaults": {"HDBK": "Office of Student Affairs and Services"},
        "rules": [
            {"doc": "HDBK", "root": "PART 3. STUDENT SUCCESS",
             "chapter": "CHAPTER IX LIBRARY SERVICES", "office": "University Library"},
        ],
    }
    office = resolve_issuing_office(
        "HDBK-0150",
        "PART 3. STUDENT SUCCESS > CHAPTER IX LIBRARY SERVICES > Section 1. General Policies (pp. 61-62)",
        table,
    )
    assert office == "University Library"


def test_resolve_falls_back_to_document_default_for_front_matter():
    table = {"document_defaults": {"INTN": "Placement, Alumni and Linkages Office"}, "rules": []}
    assert resolve_issuing_office("INTN-0001", "p. 4", table) == "Placement, Alumni and Linkages Office"


def test_every_passage_has_a_known_office():
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        passages = json.load(f)["passages"]
    table = load_office_table()
    vocabulary = set(table["office_vocabulary"])

    assert len(passages) == 364
    unknown = [p["document_id"] for p in passages if p.get("issuing_office") in (None, "", "Unknown")]
    assert not unknown, f"{len(unknown)} passages have no office: {unknown[:10]}"
    off_vocab = {p["issuing_office"] for p in passages} - vocabulary
    assert not off_vocab, f"offices outside the closed vocabulary: {off_vocab}"


def test_passage_loads_a_pre_migration_record():
    # issuing_office is defaulted, so an old corpus.json must still load.
    p = Passage(
        document_id="HDBK-0001", title="t", source="s",
        revision_date="not available", page_or_section="p. 4", text="x",
    )
    assert p.issuing_office == "Unknown"
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_corpus_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_pipeline_office'`

- [ ] **Step 4: Author `rag_pipeline/data/office_lookup.json`**

Use the skeleton in §3.2. Enumerate the real keys first so no section is missed:
```bash
cd routing && ../venv/bin/python -c "
import json, re, collections
ps = json.load(open('data/corpus.json'))['passages']
keys = collections.Counter()
for p in ps:
    bc = re.sub(r'\s*\((?:p\.|pp\.)[^)]*\)\s*$', '', p['page_or_section'])
    parts = [x.strip() for x in bc.split('>')]
    keys[(p['document_id'][:4], parts[0], parts[1] if len(parts) > 1 else parts[0])] += 1
for k, n in sorted(keys.items()):
    print(n, k)
print('DISTINCT KEYS:', len(keys))
"
```
Expect 73 keys, 19 of them bare `p. N` front matter. Write one `rules` row per substantive key (~54) and let `document_defaults` cover the rest. Keep `office_vocabulary` closed — every `office` value must appear in it.

- [ ] **Step 5: Implement the resolver**

Create `routing/rag_pipeline_office.py` (kept on the routing side so both the build script and the tests import one implementation):
```python
"""Objective 3's section -> issuing_office lookup, applied at corpus-build time.

The mapping is a researcher-constructed, versioned artifact: it makes the
manuals' existing organization-by-responsible-unit explicit. It is not
extracted from the source text (text-mining was tested and rejected: only
32% of passages mention anything office-like, and keyword matching pulls
false positives such as "Patent Office" from the IP chapter).
"""

import json
import os
import re

_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rag_pipeline", "data", "office_lookup.json",
)

_PAGE_SUFFIX = re.compile(r"\s*\((?:p\.|pp\.)[^)]*\)\s*$")


def load_office_table(path: str = _TABLE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_breadcrumb(page_or_section: str) -> list:
    """'A > B > C (pp. 4-5)' -> ['A', 'B', 'C']. Bare 'p. 4' -> ['p. 4']."""
    return [part.strip() for part in _PAGE_SUFFIX.sub("", page_or_section).split(">")]


def resolve_issuing_office(document_id: str, page_or_section: str, table: dict) -> str:
    doc = document_id[:4]
    parts = parse_breadcrumb(page_or_section)
    root = parts[0]
    chapter = parts[1] if len(parts) > 1 else parts[0]

    for rule in table.get("rules", []):
        if rule["doc"] == doc and rule["root"] == root and rule["chapter"] == chapter:
            return rule["office"]
    for rule in table.get("rules", []):
        if rule["doc"] == doc and rule["root"] == root:
            return rule["office"]
    return table.get("document_defaults", {}).get(doc, "Unknown")
```

- [ ] **Step 6: Add the field to `Passage`**

In `routing/retriever.py`, add to the dataclass — **defaulted, and last**, so it stays keyword-compatible with a pre-migration `corpus.json`:
```python
@dataclass
class Passage:
    document_id: str
    title: str
    source: str
    revision_date: str
    page_or_section: str
    text: str
    issuing_office: str = "Unknown"
```

- [ ] **Step 7: Stamp the field at build time**

In `rag_pipeline/build_corpus.py`, import the resolver and set `issuing_office` on each passage record before writing `corpus.json`. Load the table once outside the loop. Raise a clear error if any passage resolves to `"Unknown"` — a silent `"Unknown"` becomes an unusable referral at inference.

- [ ] **Step 8: Rebuild the corpus and run the tests**

Run:
```bash
../venv/bin/python rag_pipeline/build_corpus.py
cd routing && ../venv/bin/python -m pytest tests/test_corpus_schema.py -v
```
Expected: PASS, 4 tests, 364 passages all carrying a vocabulary office.

- [ ] **Step 9: Commit**

```bash
git add rag_pipeline/data/office_lookup.json rag_pipeline/build_corpus.py \
        routing/retriever.py routing/rag_pipeline_office.py \
        routing/tests/conftest.py routing/tests/test_corpus_schema.py \
        routing/requirements.txt routing/data/corpus.json
git commit -m "feat(obj5): add issuing_office to corpus via versioned office lookup table"
```

---

### Task 2: `gate2.py` — the sufficiency decision

**Files:**
- Create: `routing/gate2.py`
- Modify: `routing/config.py`
- Test: `routing/tests/test_gate2_sufficiency.py`

**Interfaces:**
- Consumes: Objective 4's retrieval output (BM25 scores + `idf`/`k1` from `BM25Okapi`, dense cosines). Nothing here retrieves.
- Produces: `Gate2Signals` with fields `coverage: float`, `norm_bm25: float`, `dense_max_cos: float`, `bm25_ranking: List[Tuple[int, float]]`, `dense_ranking: List[Tuple[int, float]]`; and `Gate2Signals.evaluate(c, s, d) -> Gate2Decision(sufficient, lexical_ok, dense_ok)`. `calibrate.py` and `router.py` both consume this.

- [ ] **Step 1: Add the tunables to `config.py`**

```python
# --- Gate 2: sufficiency (Objective 5 reads Objective 4's retrieval) ---
BM25_COVERAGE_THRESHOLD = 1.0   # c -- PLACEHOLDER, calibrated in Task 8
BM25_SCORE_THRESHOLD = 0.0      # s -- normalized top-1, PLACEHOLDER
DENSE_SCORE_THRESHOLD = 0.65    # d -- PLACEHOLDER
DENSE_TOP_N = 10
BM25_TOP_N = 10

# --- Calibration sweep grids (Task 8) ---
SWEEP_COVERAGE = [round(0.1 * i, 2) for i in range(9)]        # 0.0 .. 0.8
SWEEP_BM25_SCORE = [round(0.05 * i, 2) for i in range(11)]    # 0.0 .. 0.50
SWEEP_DENSE = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30 .. 0.80
SWEEP_SCOPE = [round(0.30 + 0.05 * i, 2) for i in range(12)]  # 0.30 .. 0.85
MAX_FALSE_ACCEPT_RATE = 0.10    # pre-registered constraint, section 2.1
CV_FOLDS = 5
```

- [ ] **Step 2: Write the failing test**

Create `routing/tests/test_gate2_sufficiency.py`:
```python
from gate2 import Gate2Signals, normalized_bm25, term_coverage


def test_normalized_bm25_is_bounded_and_scale_free():
    # Same relative match quality, different query lengths -> comparable values.
    short = normalized_bm25(raw_top1=8.0, idf_sum=6.0, k1=1.5)
    long = normalized_bm25(raw_top1=24.0, idf_sum=18.0, k1=1.5)
    assert 0.0 <= short <= 1.0
    assert abs(short - long) < 1e-9


def test_normalized_bm25_handles_all_oov_query():
    assert normalized_bm25(raw_top1=0.0, idf_sum=0.0, k1=1.5) == 0.0


def test_term_coverage_counts_matched_query_terms():
    assert term_coverage(["leave", "absenc"], "appli for a leav of absenc form") == 0.5
    assert term_coverage([], "anything") == 0.0


def test_disjunctive_gate_fires_on_dense_alone():
    sig = Gate2Signals(coverage=0.0, norm_bm25=0.01, dense_max_cos=0.80,
                       bm25_ranking=[], dense_ranking=[(3, 0.80)])
    decision = sig.evaluate(c=0.5, s=0.20, d=0.65)
    assert decision.sufficient is True
    assert decision.lexical_ok is False and decision.dense_ok is True


def test_disjunctive_gate_fires_on_lexical_alone():
    sig = Gate2Signals(coverage=0.9, norm_bm25=0.05, dense_max_cos=0.20,
                       bm25_ranking=[(1, 12.0)], dense_ranking=[])
    decision = sig.evaluate(c=0.5, s=0.20, d=0.65)
    assert decision.sufficient is True
    assert decision.lexical_ok is True and decision.dense_ok is False


def test_gate_refers_only_when_both_components_fail():
    sig = Gate2Signals(coverage=0.1, norm_bm25=0.02, dense_max_cos=0.30,
                       bm25_ranking=[(1, 3.0)], dense_ranking=[(1, 0.30)])
    assert sig.evaluate(c=0.5, s=0.20, d=0.65).sufficient is False
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_gate2_sufficiency.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gate2'`

- [ ] **Step 4: Implement `gate2.py`**

```python
"""Gate 2: evidence sufficiency.

Reads Objective 4's hybrid retrieval output and answers one question --
is there enough evidence to answer from? -- with a DISJUNCTIVE component
gate: sufficient = lexical_ok OR dense_ok.

The disjunction is deliberate. BM25 misses paraphrase; dense misses rare
exact tokens (R.A. 10627, form codes, section numbers). Referring only when
BOTH agree there is nothing is the actual definition of insufficient
evidence. Accepted cost: either retriever alone can admit a weak match,
backstopped by the prompt contract and citation validation.

Sufficiency is judged on COMPONENT scores, before fusion, and never on the
RRF score. RRF is rank-only, so the top hit for "best pizza recipe" earns
the identical fused score as the top hit for a perfect query --
thresholding it would destroy the referral route entirely.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Gate2Decision:
    sufficient: bool
    lexical_ok: bool
    dense_ok: bool


@dataclass
class Gate2Signals:
    """Everything Objective 5 needs from Objective 4's retrieval, per query."""
    coverage: float
    norm_bm25: float
    dense_max_cos: float
    bm25_ranking: List[Tuple[int, float]] = field(default_factory=list)
    dense_ranking: List[Tuple[int, float]] = field(default_factory=list)

    def evaluate(self, c: float, s: float, d: float) -> Gate2Decision:
        lexical_ok = (self.coverage >= c) or (self.norm_bm25 >= s)
        dense_ok = self.dense_max_cos >= d
        return Gate2Decision(
            sufficient=lexical_ok or dense_ok,
            lexical_ok=lexical_ok,
            dense_ok=dense_ok,
        )


def normalized_bm25(raw_top1: float, idf_sum: float, k1: float) -> float:
    """Top-1 BM25 divided by the query's own saturation ceiling.

    (k1 + 1) * sum(idf) is the score a passage would earn if every query
    term matched with unbounded term frequency, so the result is in [0, 1]
    and comparable across query lengths -- which raw BM25 is not. An
    all-OOV query has idf_sum == 0 and yields 0.0 rather than dividing by
    zero.
    """
    ceiling = (k1 + 1.0) * idf_sum
    if ceiling <= 0.0:
        return 0.0
    return min(1.0, raw_top1 / ceiling)


def term_coverage(query_terms: List[str], passage_text: str) -> float:
    """Fraction of the query's content terms present in the top-1 passage.

    Both sides must already be tokenized/stemmed by Objective 4's tokenizer
    (regex + stopwords + Snowball) -- passing raw text here silently
    understates coverage.
    """
    if not query_terms:
        return 0.0
    passage_terms = set(passage_text.split())
    return sum(1 for t in query_terms if t in passage_terms) / len(query_terms)
```

- [ ] **Step 5: Run the tests**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_gate2_sufficiency.py -v`
Expected: PASS, 6 tests

- [ ] **Step 6: Commit**

```bash
git add routing/gate2.py routing/config.py routing/tests/test_gate2_sufficiency.py
git commit -m "feat(obj5): add Gate 2 disjunctive sufficiency decision over Objective 4 signals"
```

---

### Task 3: `scope_gate.py` — labeled 1-NN scope rule

**Files:**
- Modify: `routing/scope_gate.py`
- Test: `routing/tests/test_scope_gate.py`

**Interfaces:**
- Consumes: `routing/data/scope_exemplars.json` (v2 or legacy shape); optionally a precomputed query vector
- Produces: `ScopeDecision(in_scope: bool, best_similarity: float, best_match_query: str, best_match_label: str, latency_ms: float)`; `ScopeGate.check(query, query_embedding=None)`

- [ ] **Step 1: Write the failing test**

Create `routing/tests/test_scope_gate.py`:
```python
import json

import numpy as np
import pytest

from scope_gate import ScopeGate, normalize_exemplars


def test_normalize_accepts_legacy_shape():
    legacy = {"in_scope_examples": ["How do I request a leave of absence?"]}
    out = normalize_exemplars(legacy)
    assert out == [{"query": "How do I request a leave of absence?", "label": "in_scope"}]


def test_normalize_accepts_v2_shape():
    v2 = {"version": 2, "exemplars": [
        {"query": "What is the capital of Japan?", "label": "out_of_scope",
         "family": "out_of_scope", "topic": "general_knowledge"},
    ]}
    out = normalize_exemplars(v2)
    assert out[0]["label"] == "out_of_scope"


@pytest.fixture
def gate(tmp_path):
    path = tmp_path / "exemplars.json"
    path.write_text(json.dumps({"version": 2, "exemplars": [
        {"query": "How do I borrow a book from the university library?", "label": "in_scope"},
        {"query": "How much is the fee to replace a lost student ID?", "label": "in_scope"},
        {"query": "What is the best way to cook chicken adobo?", "label": "out_of_scope"},
        {"query": "Write me a Python function to sort a list.", "label": "out_of_scope"},
    ]}), encoding="utf-8")
    return ScopeGate(exemplars_path=str(path), threshold=0.40)


def test_nearest_in_scope_exemplar_routes_in_scope(gate):
    d = gate.check("Can I borrow books from the library?")
    assert d.in_scope is True and d.best_match_label == "in_scope"


def test_nearest_out_of_scope_exemplar_routes_out_of_scope(gate):
    d = gate.check("What is a good recipe for pork sinigang?")
    assert d.in_scope is False and d.best_match_label == "out_of_scope"


def test_far_from_everything_is_out_of_scope_even_if_nearest_is_in_scope(gate):
    # The distance floor, not the label, is what rejects here.
    high = ScopeGate(exemplars_path=gate.exemplars_path, threshold=0.99)
    assert high.check("Can I borrow books from the library?").in_scope is False


def test_precomputed_embedding_is_reused_not_recomputed(gate):
    vec = gate.embed_query("Can I borrow books from the library?")
    a = gate.check("Can I borrow books from the library?")
    b = gate.check("Can I borrow books from the library?", query_embedding=vec)
    assert np.isclose(a.best_similarity, b.best_similarity)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_scope_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_exemplars'`

- [ ] **Step 3: Implement the changes**

In `routing/scope_gate.py`:

1. Add `normalize_exemplars(raw: dict) -> List[dict]` handling both shapes — v2's `exemplars` list passes through (defaulting a missing `label` to `"in_scope"`); legacy `in_scope_examples` maps each string to `{"query": s, "label": "in_scope"}`.
2. Store `self.exemplar_labels` alongside `self.exemplars` and pre-embed the query strings exactly as today.
3. Add `ScopeGate.embed_query(query) -> np.ndarray` applying the `bge` **query-side instruction prefix** (`"Represent this sentence for searching relevant passages: "`) — passages must not receive it. Getting this backwards degrades retrieval quietly, so it belongs in one place both gates call.
4. Change `check()` to accept `query_embedding: Optional[np.ndarray] = None`, embedding only when it is `None`, and to apply the labeled rule:
```python
in_scope = (best_label == "in_scope") and (best_similarity >= self.threshold)
```
5. Add `best_match_label: str` to `ScopeDecision` **as the fourth field — after `best_match_query`, before `latency_ms`** (the tests construct it positionally), and keep `exemplars_path` as an attribute (the tests reuse it):
```python
@dataclass
class ScopeDecision:
    in_scope: bool
    best_similarity: float
    best_match_query: str
    best_match_label: str
    latency_ms: float
```

- [ ] **Step 4: Run the tests**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_scope_gate.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add routing/scope_gate.py routing/tests/test_scope_gate.py
git commit -m "feat(obj5): labeled 1-NN scope rule with shared query embedding"
```

---

### Task 4: Author the Gate 1 exemplar set

**Files:**
- Replace: `routing/data/scope_exemplars.json`
- Test: `routing/tests/test_exemplar_coverage.py`

**Interfaces:**
- Consumes: `routing/data/corpus.json` (for chapter coverage)
- Produces: v2-shaped exemplar file, ~65 entries, consumed by `ScopeGate`

> **This is authoring work, not coding.** The current file's 20 generic US-university strings do not describe the UC-PnC corpus and carry no out-of-scope half. Budget ~1 hour.

- [ ] **Step 1: Write the failing coverage test**

Create `routing/tests/test_exemplar_coverage.py`:
```python
import json
import re
from collections import Counter

import config
from scope_gate import normalize_exemplars

_PAGE_SUFFIX = re.compile(r"\s*\((?:p\.|pp\.)[^)]*\)\s*$")


def _load():
    with open(config.SCOPE_EXEMPLARS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_has_both_labels_and_both_in_scope_families():
    raw = _load()
    fams = Counter((e["label"], e.get("family", "")) for e in raw["exemplars"])
    labels = Counter(e["label"] for e in raw["exemplars"])

    assert labels["in_scope"] >= 40, labels
    assert labels["out_of_scope"] >= 15, labels
    # The family that makes the REFER path reachable at all.
    assert fams[("in_scope", "university_not_in_corpus")] >= 12, fams
    assert fams[("in_scope", "corpus_topic")] >= 25, fams


def test_out_of_scope_spans_five_categories():
    raw = _load()
    topics = Counter(e["topic"] for e in raw["exemplars"] if e["label"] == "out_of_scope")
    required = {"general_knowledge", "other_institution", "chit_chat",
                "coding_homework", "local_services"}
    assert required <= set(topics), required - set(topics)
    assert all(topics[t] >= 3 for t in required), topics


def test_corpus_chapters_have_an_in_scope_neighbour():
    """Every substantive chapter should be represented by >=1 corpus_topic exemplar topic."""
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        passages = json.load(f)["passages"]
    chapters = set()
    for p in passages:
        parts = [x.strip() for x in _PAGE_SUFFIX.sub("", p["page_or_section"]).split(">")]
        if len(parts) > 1 and not parts[0].startswith("p. "):
            chapters.add(parts[1])

    topics = {e["topic"] for e in _load()["exemplars"]
              if e.get("family") == "corpus_topic"}
    # One topic may serve several chapters; require breadth, not 1:1.
    assert len(topics) >= 15, sorted(topics)
    assert len(chapters) > 0


def test_normalize_round_trips_the_authored_file():
    assert len(normalize_exemplars(_load())) >= 55
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_exemplar_coverage.py -v`
Expected: FAIL — the current file has `in_scope_examples`, no `exemplars` key

- [ ] **Step 3: Enumerate the chapters to cover**

```bash
cd routing && ../venv/bin/python -c "
import json, re
ps = json.load(open('data/corpus.json'))['passages']
seen = {}
for p in ps:
    parts = [x.strip() for x in re.sub(r'\s*\((?:p\.|pp\.)[^)]*\)\s*$','',p['page_or_section']).split('>')]
    if len(parts) > 1 and not parts[0].startswith('p. '):
        seen.setdefault((p['document_id'][:4], parts[1]), 0)
        seen[(p['document_id'][:4], parts[1])] += 1
for k, n in sorted(seen.items()): print(n, k)
"
```

- [ ] **Step 4: Author the file**

Write `routing/data/scope_exemplars.json` in the §3.4 shape:
- **~30 `corpus_topic`** in-scope exemplars — one per substantive chapter a student could ask about, phrased the way a student asks ("Can I retake a subject I failed?"), not the way the manual is titled. All three manuals represented.
- **~15 `university_not_in_corpus`** in-scope exemplars — dormitory/housing, ID replacement fee, shuttle service, canteen, tuition instalment schedules, campus WiFi, uniform purchase, lost-and-found, enrolment payment channels. **Do not skip this family.** Without it Gate 1 sends the entire 25-query referral stratum to `out_of_scope` and Path B never fires.
- **~20 `out_of_scope`**, 4 each across `general_knowledge`, `other_institution`, `chit_chat`, `coding_homework`, `local_services`.

- [ ] **Step 5: Run the tests**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_exemplar_coverage.py -v`
Expected: PASS, 4 tests

- [ ] **Step 6: Commit**

```bash
git add routing/data/scope_exemplars.json routing/tests/test_exemplar_coverage.py
git commit -m "feat(obj5): replace generic exemplars with labeled UC-PnC scope exemplar set"
```

---

### Task 5: `referral.py` — office selection with deterministic override

**Files:**
- Create: `routing/referral.py`
- Modify: `routing/config.py`
- Test: `routing/tests/test_referral.py`

**Interfaces:**
- Consumes: `Gate2Signals.bm25_ranking` / `dense_ranking`, `List[Passage]`, `config.RRF_K`
- Produces: `rank_office_candidates(bm25_ranking, dense_ranking, passages, k, max_candidates) -> List[str]`; `select_office(model_line, candidates) -> Tuple[str, bool]`; `validate_phrasing(sentence, office) -> bool`; `ReferralResult(office, message, used_model, overridden, latency_ms)`; `ReferralAgent.refer(query, candidates) -> ReferralResult`

- [ ] **Step 1: Add referral config**

```python
# --- Path B: office referral (Objective 5) ---
REFERRAL_MAX_CANDIDATES = 3
REFERRAL_USE_MODEL_PHRASING = True   # v2: model phrases the sentence; validated before display
REFERRAL_MAX_TOKENS = 80

REFERRAL_SYSTEM_PROMPT = """You route university helpdesk questions to the correct office.
You are given a student's question and a numbered list of candidate offices.
Choose exactly one office from the list and write one short sentence telling the student to contact it.
Never name an office that is not in the list. Never answer the question itself.
Never state contact details, room numbers, hours, fees, deadlines, or web addresses.
Reply with the office name on the first line and the sentence on the second line. Nothing else."""

REFERRAL_TEMPLATE = (
    "I could not verify this from the selected LMS materials. "
    "Please contact the {office} for this concern."
)
```

- [ ] **Step 2: Write the failing test**

Create `routing/tests/test_referral.py`:
```python
from referral import rank_office_candidates, select_office, validate_phrasing
from retriever import Passage


def _passages():
    return [
        Passage("HDBK-0001", "t", "s", "not available", "p. 4", "x", "University Library"),
        Passage("HDBK-0002", "t", "s", "not available", "p. 5", "x", "Guidance and Counseling Office"),
        Passage("HDBK-0003", "t", "s", "not available", "p. 6", "x", "University Library"),
        Passage("HDBK-0004", "t", "s", "not available", "p. 7", "x", "Office of the University Registrar"),
        Passage("HDBK-0005", "t", "s", "not available", "p. 8", "x", "Unknown"),
    ]


def test_candidates_are_deduped_rank_ordered_and_capped():
    # Library appears at ranks 1 and 3; it must appear once, first.
    out = rank_office_candidates(
        bm25_ranking=[(0, 9.0), (2, 8.0), (1, 7.0), (3, 6.0)],
        dense_ranking=[(0, 0.7), (1, 0.6), (3, 0.5), (2, 0.4)],
        passages=_passages(), k=10, max_candidates=3,
    )
    assert out[0] == "University Library"
    assert len(out) == len(set(out)) <= 3


def test_unknown_offices_are_dropped():
    out = rank_office_candidates(
        bm25_ranking=[(4, 9.0), (0, 8.0)], dense_ranking=[(4, 0.9), (0, 0.8)],
        passages=_passages(), k=10, max_candidates=3,
    )
    assert "Unknown" not in out


def test_empty_rankings_yield_no_candidates():
    assert rank_office_candidates([], [], _passages(), k=10, max_candidates=3) == []


CANDIDATES = ["University Library", "Guidance and Counseling Office",
              "Office of the University Registrar"]


def test_exact_match_is_accepted_without_override():
    office, overridden = select_office("Guidance and Counseling Office", CANDIDATES)
    assert office == "Guidance and Counseling Office" and overridden is False


def test_match_is_case_and_whitespace_insensitive():
    office, overridden = select_office("  guidance   and counseling office ", CANDIDATES)
    assert office == "Guidance and Counseling Office" and overridden is False


def test_off_list_office_is_overridden_to_top_candidate():
    office, overridden = select_office("Office of the Dean of Wizardry", CANDIDATES)
    assert office == "University Library" and overridden is True


def test_ambiguous_multi_match_is_overridden():
    office, overridden = select_office(
        "University Library or Office of the University Registrar", CANDIDATES)
    assert office == "University Library" and overridden is True


def test_out_of_range_index_is_overridden():
    office, overridden = select_office("7. Some Office", CANDIDATES)
    assert office == "University Library" and overridden is True


def test_phrasing_validation_rejects_invented_contact_details():
    office = "University Library"
    assert validate_phrasing("Please contact the University Library for help.", office) is True
    assert validate_phrasing("Call the University Library at 049-555-1234.", office) is False
    assert validate_phrasing("Email libraryuc@example.com.", office) is False
    assert validate_phrasing("See http://uc.edu.ph/library", office) is False
    assert validate_phrasing("Please contact the registrar.", office) is False
    assert validate_phrasing("x" * 201 + " University Library", office) is False
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_referral.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'referral'`

- [ ] **Step 4: Implement `referral.py`**

```python
"""Path B: office referral.

Bounds the model's job to picking one name from a 2-3 item list and
phrasing one sentence -- never open-ended fact invention. The prompt
carries no passages, only the question and a few short office names, so
token cost stays in the same class as a single short generation.

Two safety nets:
  * select_office() overrides deterministically to the top-ranked
    candidate whenever the model names something off-list, names two
    things, or emits an out-of-range index.
  * validate_phrasing() rejects any sentence containing a digit, '@', or
    'http' -- exactly the characters an invented phone number, email,
    room number, or URL would arrive in -- falling back to a template.
"""

import re
import time
from dataclasses import dataclass
from typing import List, Tuple

from openai import OpenAI

import config
from retriever import Passage

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().casefold()


@dataclass
class ReferralResult:
    office: str
    message: str
    used_model: bool
    overridden: bool
    latency_ms: float = 0.0


def rank_office_candidates(
    bm25_ranking: List[Tuple[int, float]],
    dense_ranking: List[Tuple[int, float]],
    passages: List[Passage],
    k: int,
    max_candidates: int,
) -> List[str]:
    """Referral-only RRF (decision D-1).

    Neither retriever's gate fired, so this ranking is never used as
    evidence and never influences sufficiency -- it only decides WHICH
    office name to put in front of the student. Fusing here costs nothing:
    both candidate lists are already computed.
    """
    fused = {}
    for ranking in (bm25_ranking, dense_ranking):
        for rank, (idx, _score) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    offices: List[str] = []
    for idx, _ in ordered:
        office = passages[idx].issuing_office
        if office and office != "Unknown" and office not in offices:
            offices.append(office)
        if len(offices) >= max_candidates:
            break
    return offices


def select_office(model_line: str, candidates: List[str]) -> Tuple[str, bool]:
    """Return (office, overridden). Overrides to candidates[0] on any doubt."""
    if not candidates:
        return "", False

    line = _norm(model_line)
    matched = [c for c in candidates if _norm(c) in line]

    if len(matched) == 1:
        return matched[0], False

    if not matched:
        leading = re.match(r"^\s*(\d+)", model_line)
        if leading:
            i = int(leading.group(1))
            if 1 <= i <= len(candidates):
                return candidates[i - 1], False

    return candidates[0], True


def validate_phrasing(sentence: str, office: str) -> bool:
    """Model phrasing is displayed only if it survives this."""
    if not sentence or len(sentence) > 200:
        return False
    if _norm(office) not in _norm(sentence):
        return False
    if any(ch.isdigit() for ch in sentence):
        return False
    if "@" in sentence or "http" in sentence.lower():
        return False
    return True


class ReferralAgent:
    def __init__(self, model: str = config.GENERATOR_MODEL):
        self.model = model
        self._client = None  # lazy: the 0- and 1-candidate paths need no key

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not config.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set; needed for the referral call.")
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    def refer(self, query: str, candidates: List[str]) -> ReferralResult:
        start = time.perf_counter()

        if not candidates:
            return ReferralResult(
                office="", message=config.REFERRAL_MESSAGE,
                used_model=False, overridden=False,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        if len(candidates) == 1:
            office = candidates[0]
            return ReferralResult(
                office=office, message=config.REFERRAL_TEMPLATE.format(office=office),
                used_model=False, overridden=False,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        listing = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates, start=1))
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=config.GENERATOR_TEMPERATURE,
            max_tokens=config.REFERRAL_MAX_TOKENS,
            messages=[
                {"role": "system", "content": config.REFERRAL_SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"Question: {query}\n\nCandidate offices:\n{listing}"},
            ],
        )

        lines = [ln for ln in (response.choices[0].message.content or "").splitlines() if ln.strip()]
        office, overridden = select_office(lines[0] if lines else "", candidates)

        message = config.REFERRAL_TEMPLATE.format(office=office)
        if config.REFERRAL_USE_MODEL_PHRASING and len(lines) > 1:
            if validate_phrasing(lines[1].strip(), office):
                message = lines[1].strip()

        return ReferralResult(
            office=office, message=message, used_model=True, overridden=overridden,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
```

- [ ] **Step 5: Run the tests**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_referral.py -v`
Expected: PASS, 10 tests. No network call is made — every test exercises the pure functions.

- [ ] **Step 6: Commit**

```bash
git add routing/referral.py routing/config.py routing/tests/test_referral.py
git commit -m "feat(obj5): office referral with deterministic override and phrasing validation"
```

---

### Task 6: `router.py` — wire the three paths

**Files:**
- Modify: `routing/router.py`
- Modify: `routing/main.py`
- Test: `routing/tests/test_router_paths.py`

**Interfaces:**
- Consumes: `Gate2Signals`, `ScopeGate.check`, `ReferralAgent.refer`, `RAGGenerator.generate`
- Produces: `RouteResult` extended with `referral_office: Optional[str]`, `referral_candidates: List[str]`, `referral_overridden: bool`, `gate2_lexical_ok: bool`, `gate2_dense_ok: bool`, `norm_bm25: float`, `dense_max_cos: float`, `coverage: float`, `referral_latency_ms: float`

- [ ] **Step 1: Write the failing test**

Create `routing/tests/test_router_paths.py`:
```python
from unittest.mock import MagicMock

import config
from gate2 import Gate2Signals
from referral import ReferralResult
from router import Router
from scope_gate import ScopeDecision


def _router(signals, scope, referral=None, answer="An answer. [HDBK-0001]"):
    r = Router.__new__(Router)                       # bypass real model loading
    r.retriever = MagicMock()
    r.retriever.signals_for.return_value = signals
    r.retriever.passages_for.return_value = []
    r.scope_gate = MagicMock()
    r.scope_gate.check.return_value = scope
    r.scope_gate.embed_query.return_value = None
    r.generator = MagicMock()
    r.generator.generate.return_value = MagicMock(
        answer=answer, citations=["HDBK-0001"], latency_ms=1.0)
    r.referral = MagicMock()
    r.referral.refer.return_value = referral or ReferralResult(
        "University Library", "Contact the University Library.", True, False, 1.0)
    return r


SUFFICIENT = Gate2Signals(coverage=0.9, norm_bm25=0.4, dense_max_cos=0.9,
                          bm25_ranking=[(0, 9.0)], dense_ranking=[(0, 0.9)])
INSUFFICIENT = Gate2Signals(coverage=0.0, norm_bm25=0.0, dense_max_cos=0.1,
                            bm25_ranking=[(0, 1.0)], dense_ranking=[(0, 0.1)])

IN_SCOPE = ScopeDecision(True, 0.80, "exemplar", "in_scope", 1.0)
OUT_OF_SCOPE = ScopeDecision(False, 0.10, "exemplar", "out_of_scope", 1.0)


def test_path_a_sufficient_evidence_answers():
    result = _router(SUFFICIENT, IN_SCOPE).route("q")
    assert result.decision == "answer"
    assert result.generator_calls == 1


def test_path_b_insufficient_and_in_scope_refers_with_an_office():
    result = _router(INSUFFICIENT, IN_SCOPE).route("q")
    assert result.decision == "refer"
    assert result.referral_office == "University Library"
    assert result.generator_calls == 1          # the short referral call


def test_path_c_insufficient_and_out_of_scope_is_free_and_static():
    result = _router(INSUFFICIENT, OUT_OF_SCOPE).route("q")
    assert result.decision == "out_of_scope"
    assert result.answer == config.SCOPE_MESSAGE
    assert result.generator_calls == 0


def test_referral_without_candidates_makes_no_model_call():
    referral = ReferralResult("", config.REFERRAL_MESSAGE, False, False, 0.5)
    result = _router(INSUFFICIENT, IN_SCOPE, referral=referral).route("q")
    assert result.decision == "refer"
    assert result.generator_calls == 0


def test_ORDERING_INVARIANT_sufficient_evidence_is_never_out_of_scope():
    """V4. This test IS the false-out-of-scope failure mode, encoded.

    A paraphrased in-scope query ("can I take a break?" vs "Leave of
    Absence") can score arbitrarily low against the exemplars. If Gate 2
    found evidence, Gate 1's verdict must be unreachable.
    """
    result = _router(SUFFICIENT, OUT_OF_SCOPE).route("q")
    assert result.decision == "answer"
    assert result.scope_in_scope is False       # computed and logged...
    assert result.scope_similarity == 0.10      # ...but never acted on


def test_scope_similarity_is_logged_on_every_path():
    for signals, scope in ((SUFFICIENT, IN_SCOPE), (INSUFFICIENT, IN_SCOPE),
                           (INSUFFICIENT, OUT_OF_SCOPE)):
        assert _router(signals, scope).route("q").scope_similarity is not None
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_router_paths.py -v`
Expected: FAIL — `RouteResult` has no `referral_office`; `Router` has no `referral`

- [ ] **Step 3: Rewrite `Router.route()`**

Keep the existing docstring and per-gate latency fields. Structure:

```python
def route(self, query: str) -> RouteResult:
    total_start = time.perf_counter()

    query_embedding = self.scope_gate.embed_query(query)      # embed ONCE

    signals = self.retriever.signals_for(query, query_embedding)   # Gate 2 (Objective 4)
    gate2 = signals.evaluate(
        c=config.BM25_COVERAGE_THRESHOLD,
        s=config.BM25_SCORE_THRESHOLD,
        d=config.DENSE_SCORE_THRESHOLD,
    )

    # Gate 1 runs on EVERY query so scope_similarity is always logged --
    # but below it is only reachable when Gate 2 came up short.
    scope = self.scope_gate.check(query, query_embedding=query_embedding)

    if gate2.sufficient:
        ...   # PATH A: fuse the lists whose gate fired -> Gate 3 -> decision="answer"

    if not scope.in_scope:
        ...   # PATH C: static SCOPE_MESSAGE, generator_calls=0

    # PATH B
    candidates = rank_office_candidates(
        signals.bm25_ranking, signals.dense_ranking,
        self.retriever.passages_for(signals), config.RRF_K,
        config.REFERRAL_MAX_CANDIDATES,
    )
    referral = self.referral.refer(query, candidates)
    ...   # decision="refer", generator_calls = 1 if referral.used_model else 0
```

Populate the new `RouteResult` fields on every path (use `None` / `[]` / `False` where a path does not apply) so the eval harness never has to special-case a missing attribute.

- [ ] **Step 4: Run the tests**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_router_paths.py -v`
Expected: PASS, 6 tests — including the V4 ordering invariant

- [ ] **Step 5: Update the demo printout**

In `routing/main.py`, add `referral_office`, `referral_candidates`, `referral_overridden`, `norm_bm25`, `coverage`, and `dense_max_cos` to `print_result`, and update `SAMPLE_QUERIES` to UC-PnC content (the current samples are the same generic US-university phrasing as the old exemplars).

- [ ] **Step 6: Commit**

```bash
git add routing/router.py routing/main.py routing/tests/test_router_paths.py
git commit -m "feat(obj5): wire three-path routing with referral and ordering invariant test"
```

---

### Task 7: Author the 80-query eval set

**Files:**
- Create: `routing/data/eval_queries.json`
- Test: `routing/tests/test_eval_data.py`

**Interfaces:**
- Consumes: `corpus.json`, `office_lookup.json`, `scope_exemplars.json`
- Produces: the labeled set every downstream stage reads

> **User-authored.** `EVALUATION_PLAN.md` estimates days for this, and it is the long pole. Tasks 8 and 9 are blocked on it; Tasks 1–6 are not, so this can be authored in parallel with them.

- [ ] **Step 1: Write the failing validation test**

Create `routing/tests/test_eval_data.py`:
```python
import json
from collections import Counter

import numpy as np

import config
from scope_gate import ScopeGate, normalize_exemplars

EXPECTED = {"answerable": "answer",
            "in_scope_no_evidence": "refer",
            "out_of_scope": "out_of_scope"}


def _eval():
    with open(config.EVAL_QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_strata_counts_match_the_evaluation_plan():
    strata = Counter(q["stratum"] for q in _eval()["queries"])
    assert 35 <= strata["answerable"] <= 45, strata
    assert 20 <= strata["in_scope_no_evidence"] <= 30, strata
    assert 12 <= strata["out_of_scope"] <= 18, strata
    assert 75 <= sum(strata.values()) <= 85, strata


def test_ids_unique_and_decisions_follow_the_stratum():
    queries = _eval()["queries"]
    assert len({q["query_id"] for q in queries}) == len(queries)
    for q in queries:
        assert q["expected_decision"] == EXPECTED[q["stratum"]], q["query_id"]


def test_referral_rows_name_an_office_from_the_closed_vocabulary():
    from rag_pipeline_office import load_office_table
    vocabulary = set(load_office_table()["office_vocabulary"])
    for q in _eval()["queries"]:
        if q["stratum"] == "in_scope_no_evidence":
            assert q["expected_office"] in vocabulary, (q["query_id"], q["expected_office"])
        else:
            assert q["expected_office"] is None, q["query_id"]


def test_answerable_rows_cite_real_gold_chunks():
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        known = {p["document_id"] for p in json.load(f)["passages"]}
    for q in _eval()["queries"]:
        if q["stratum"] == "answerable":
            assert q["gold_document_ids"], q["query_id"]
            assert set(q["gold_document_ids"]) <= known, q["query_id"]
        else:
            assert q["gold_document_ids"] == [], q["query_id"]


def test_every_query_has_a_paraphrase_family_for_grouped_cv():
    families = Counter(q["paraphrase_family"] for q in _eval()["queries"])
    assert all(q["paraphrase_family"] for q in _eval()["queries"])
    # A family larger than a fold would make grouped CV degenerate.
    assert max(families.values()) <= 4, families.most_common(3)


def test_no_exemplar_leaks_into_the_eval_set():
    """Thresholds calibrated on queries that ARE the exemplars measure nothing."""
    gate = ScopeGate()
    ex = [e["query"] for e in normalize_exemplars(
        json.load(open(config.SCOPE_EXEMPLARS_PATH, encoding="utf-8")))]
    ev = [q["query"] for q in _eval()["queries"]]

    ex_vecs = gate.model.encode(ex, normalize_embeddings=True, device=gate.device)
    ev_vecs = gate.model.encode(ev, normalize_embeddings=True, device=gate.device)
    sims = ex_vecs @ ev_vecs.T

    i, j = np.unravel_index(np.argmax(sims), sims.shape)
    violations = [(ex[a], ev[b], float(sims[a, b]))
                  for a, b in zip(*np.where(sims > 0.95))]
    assert not violations, f"{len(violations)} leaks, worst: {ex[i]!r} ~ {ev[j]!r} @ {sims[i, j]:.3f}"
```

- [ ] **Step 2: Add the path to `config.py`**

```python
EVAL_QUERIES_PATH = os.path.join(DATA_DIR, "eval_queries.json")
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_eval_data.py -v`
Expected: FAIL — `FileNotFoundError: eval_queries.json`

- [ ] **Step 4: Author the 80 queries**

Use the §3.3 schema. Authoring order that avoids rework:
1. **~40 `answerable`** — walk the corpus by chapter, write a student-register question per gold chunk, record `gold_document_ids`. Stratify across the three manuals roughly in proportion to their chunk counts (234 / 108 / 22). Include both exact-token questions (form codes, `R.A.` numbers, section numbers) and paraphrase questions — Gate 2's disjunction only demonstrates its value if both are present.
2. **~25 `in_scope_no_evidence`** — real UC-PnC student concerns the three manuals do not cover. For each, record the `expected_office` a human would route it to, drawn from `office_vocabulary`. **Verify absence** by searching the corpus text before labeling a query as having no evidence; a query that *is* answerable but mislabeled poisons both routing accuracy and selective risk.
3. **~15 `out_of_scope`** — the five categories from §2.2.
4. Assign `paraphrase_family` last, across the whole set, so paraphrases written hours apart still land in the same group.

- [ ] **Step 5: Run the tests**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_eval_data.py -v`
Expected: PASS, 6 tests. If the leakage test fails, rewrite the **exemplar**, never the eval query.

- [ ] **Step 6: Commit**

```bash
git add routing/data/eval_queries.json routing/config.py routing/tests/test_eval_data.py
git commit -m "feat(obj5): add 80-query labeled routing evaluation set"
```

---

### Task 8: `eval/` — harness, calibration sweep, metrics

**Files:**
- Create: `routing/eval/__init__.py`, `harness.py`, `calibrate.py`, `report.py`, `run_poc.py`
- Test: `routing/tests/test_metrics.py`

**Interfaces:**
- Consumes: `eval_queries.json`, `Gate2Signals`, `ScopeGate`
- Produces: `collect_signals() -> dict` (writes `eval/out/signals.json`); `sweep(rows, grids) -> List[SweepRow]`; `select_operating_point(sweep_rows, max_false_accept) -> OperatingPoint`; `grouped_cv(rows, folds) -> CVResult`; `routing_accuracy(rows)`, `coverage(rows)`, `selective_risk(rows)`, `confusion_matrix(rows)`

- [ ] **Step 1: Write the failing metrics test**

Create `routing/tests/test_metrics.py`:
```python
from eval.report import confusion_matrix, coverage, routing_accuracy, selective_risk

# (true_stratum, predicted_decision)
ROWS = [
    {"stratum": "answerable", "predicted": "answer"},
    {"stratum": "answerable", "predicted": "answer"},
    {"stratum": "answerable", "predicted": "refer"},
    {"stratum": "in_scope_no_evidence", "predicted": "refer"},
    {"stratum": "in_scope_no_evidence", "predicted": "answer"},
    {"stratum": "out_of_scope", "predicted": "out_of_scope"},
]


def test_routing_accuracy_is_the_correct_path_fraction():
    assert routing_accuracy(ROWS) == 4 / 6


def test_coverage_is_the_answer_routed_fraction():
    assert coverage(ROWS) == 3 / 6


def test_selective_risk_is_error_among_answered_only():
    # 3 answered; 1 of them (the in_scope_no_evidence row) was not answerable.
    assert selective_risk(ROWS) == 1 / 3


def test_selective_risk_of_a_system_that_answers_nothing_is_zero_not_nan():
    assert selective_risk([{"stratum": "answerable", "predicted": "refer"}]) == 0.0


def test_confusion_matrix_exposes_the_forbidden_cell():
    cm = confusion_matrix(ROWS)
    assert cm["answerable"]["out_of_scope"] == 0     # V4
    assert cm["answerable"]["answer"] == 2
    assert cm["in_scope_no_evidence"]["answer"] == 1
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval'`

- [ ] **Step 3: Implement `report.py`**

```python
"""The three metrics, and nothing else.

Objective 5's metric set is frozen at routing accuracy, coverage, and
selective risk. Office-selection correctness is reported as a BREAKDOWN of
routing accuracy's refer row -- never as a fourth metric.
"""

STRATUM_TO_DECISION = {
    "answerable": "answer",
    "in_scope_no_evidence": "refer",
    "out_of_scope": "out_of_scope",
}
DECISIONS = ("answer", "refer", "out_of_scope")


def routing_accuracy(rows):
    if not rows:
        return 0.0
    correct = sum(1 for r in rows
                  if r["predicted"] == STRATUM_TO_DECISION[r["stratum"]])
    return correct / len(rows)


def coverage(rows):
    """Fraction the system was willing to ATTEMPT AN ANSWER on.

    Catches a system that refuses everything to look safe. Note this is
    narrower than the Definition of Terms wording ("answer or refer/route
    on at all") -- see the assumptions section; v2's technical text is the
    one implemented here.
    """
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["predicted"] == "answer") / len(rows)


def selective_risk(rows):
    """Error rate among ONLY the queries the system chose to answer.

    Catches a system that answers too much, too carelessly. At PoC stage
    this is the ROUTING-level risk (was the answered query actually
    answerable?) and is a lower bound on the thesis figure, which also
    counts answers graded incorrect by Objectives 2 and 4.
    """
    answered = [r for r in rows if r["predicted"] == "answer"]
    if not answered:
        return 0.0
    return sum(1 for r in answered if r["stratum"] != "answerable") / len(answered)


def confusion_matrix(rows):
    cm = {s: {d: 0 for d in DECISIONS} for s in STRATUM_TO_DECISION}
    for r in rows:
        cm[r["stratum"]][r["predicted"]] += 1
    return cm


def office_breakdown(rows):
    """Refer-row breakdown: of correctly-referred queries, how many named
    the expected office? Reported under routing accuracy, not beside it."""
    referred = [r for r in rows
                if r["predicted"] == "refer" and r["stratum"] == "in_scope_no_evidence"]
    if not referred:
        return {"n": 0, "office_correct": 0.0, "override_rate": 0.0}
    return {
        "n": len(referred),
        "office_correct": sum(1 for r in referred
                              if r.get("predicted_office") == r.get("expected_office")) / len(referred),
        "override_rate": sum(1 for r in referred if r.get("overridden")) / len(referred),
    }
```

- [ ] **Step 4: Run the metrics tests**

Run: `cd routing && ../venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Implement `harness.py`**

`collect_signals()` loads `eval_queries.json`, and for each query: embeds once via `ScopeGate.embed_query`, calls Objective 4's retrieval to build `Gate2Signals`, calls `ScopeGate.check` with the precomputed vector, and calls `rank_office_candidates`. Writes one row per query to `eval/out/signals.json`:

```json
{
  "corpus_hash": "sha256:...",
  "generated": "2026-08-08T12:00:00",
  "rows": [
    {"query_id": "Q001", "stratum": "answerable", "expected_office": null,
     "paraphrase_family": "grading_computation",
     "coverage": 0.75, "norm_bm25": 0.31, "dense_max_cos": 0.78,
     "scope_max_sim": 0.62, "scope_nn_label": "in_scope",
     "office_candidates": ["Office of the University Registrar"]}
  ]
}
```

**No model call happens in this stage.** Compare `corpus_hash` against `eval_queries.json`'s and refuse to proceed on mismatch — stale labels against a rebuilt corpus produce confident, wrong numbers.

- [ ] **Step 6: Implement `calibrate.py`**

- `replay(row, c, s, d, t) -> "answer" | "refer" | "out_of_scope"` — pure arithmetic over one cached row; this is the whole router's decision logic with no I/O.
- `sweep(rows, grids)` — the full `c × s × d × t` grid (9 × 11 × 11 × 12 = 13,068 points × 80 rows). Emits per-point routing accuracy, coverage, selective risk, false-accept rate, and the confusion matrix. Write to `eval/out/sweep.csv`.
- `select_operating_point(sweep_rows, max_false_accept=config.MAX_FALSE_ACCEPT_RATE)` — implements §2.1's pre-registered rule exactly: maximize `answerable` recall subject to false-accept ≤ 0.10, tie-break toward higher `d` then lower `c`.
- `grouped_cv(rows, folds=config.CV_FOLDS)` — group by `paraphrase_family`, assign whole families to folds, select the operating point on 4 folds, evaluate on the 5th, return per-fold and mean routing accuracy.

- [ ] **Step 7: Implement `run_poc.py`**

`argparse` with `--stage {signals,sweep,report,referral}`. `report` renders `eval/out/report.md` containing: the 3×3 confusion matrix, the three metrics at the frozen point, the CV table, the office breakdown, and an explicit **PASS/FAIL line per V1–V9** so the pre-registered criteria are checked mechanically rather than by eye.

- [ ] **Step 8: Run the full test suite**

Run: `cd routing && ../venv/bin/python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 9: Commit**

```bash
git add routing/eval routing/tests/test_metrics.py
git commit -m "feat(obj5): evaluation harness, threshold sweep, grouped CV, and the three metrics"
```

---

### Task 9: Run the PoC validation and freeze the operating point

**Files:**
- Modify: `routing/config.py` (frozen thresholds)
- Create: `routing/eval/out/report.md` (committed as the record)

**Interfaces:**
- Consumes: everything above
- Produces: calibrated `BM25_COVERAGE_THRESHOLD`, `BM25_SCORE_THRESHOLD`, `DENSE_SCORE_THRESHOLD`, `SCOPE_SIMILARITY_THRESHOLD`, `RRF_K`; the V1–V9 verdict

- [ ] **Step 1: Confirm the external prerequisites**

P1 (tokenizer fix) is mandatory before any number means anything. If P2/P3 (dense + RRF) have not landed, proceed as **Pass 0** with `dense_max_cos` hardwired to `0.0`, and label every output file `PASS-0-NOT-A-RESULT`.

- [ ] **Step 2: Collect signals**

Run: `cd routing && ../venv/bin/python -m eval.run_poc --stage signals`
Expected: `eval/out/signals.json` with 80 rows, corpus hash matching, zero model calls.

- [ ] **Step 3: Check V1 before sweeping**

Print per-stratum percentiles of `max(norm_bm25, dense_max_cos)`. If the `answerable` 10th percentile does not exceed the non-answerable 90th percentile, **stop**: no operating point satisfies §2.1's rule, and sweeping will only produce a threshold that looks acceptable by accident. Report the overlap and revisit the sufficiency signal.

- [ ] **Step 4: Sweep and select**

Run: `cd routing && ../venv/bin/python -m eval.run_poc --stage sweep`
Expected: `eval/out/sweep.csv`, plus a printed operating point and its false-accept rate.

- [ ] **Step 5: Report and check V1–V9**

Run: `cd routing && ../venv/bin/python -m eval.run_poc --stage report`
Expected: `eval/out/report.md` with a PASS/FAIL line for each of V1–V9. **V4 must be PASS unconditionally** — a failure there is a code defect in `router.py`, not a tuning problem, and Task 6's invariant test should have caught it.

- [ ] **Step 6: Run the referral stage**

Run: `cd routing && ../venv/bin/python -m eval.run_poc --stage referral`
Expected: ~25 short model calls. Compare model office accuracy against deterministic `candidates[0]` (V8). If the model does not beat it, set `REFERRAL_USE_MODEL_PHRASING = False` and drop the call — record that as a deliberate simplification, with the number that justified it.

- [ ] **Step 7: Freeze the thresholds**

Write the selected values into `config.py`, replacing the placeholder comments with the date, the sweep row they came from, and the false-accept rate at that point. A frozen threshold with no provenance is indistinguishable from a guess six months later.

- [ ] **Step 8: Commit**

```bash
git add routing/config.py routing/eval/out/report.md
git commit -m "feat(obj5): freeze calibrated routing thresholds and record PoC validation results"
```

---

# SECTION 7 — ASSUMPTIONS MADE

Everywhere this plan had to decide something the Notion notes and the corpus did not settle. Each is reversible; each is flagged so it can be checked rather than inherited silently.

| # | Assumption | Basis | If wrong |
|---|---|---|---|
| **A1** | **Referral-only RRF (D-1).** Path B fuses both candidate lists to rank offices even though neither gate fired. | Notes say "top 2–3 retrieved candidates' `issuing_office`" but never say *which* ranking when both gates failed. | Fall back to BM25 rank alone. One-line change in `rank_office_candidates`. |
| **A2** | **Labeled 1-NN with a distance floor (D-2)** for Gate 1, rather than an in-scope-only threshold. | v2 says "exemplar in-scope/**out-of-scope** queries" but the shipped file has in-scope only. A labeled NN is the literal reading. | Revert to in-scope-only + threshold; keep the out-of-scope exemplars as a reported diagnostic. |
| **A3** | **`norm_bm25 = raw / ((k1+1)·Σidf)`.** The hybrid spec names `BM25_SCORE_THRESHOLD (s, normalized)` but never defines the normalization. | Raw BM25 does not transfer across query lengths; this ceiling is query-specific and bounded. | Any monotone per-query normalization substitutes; the sweep re-fits `s` automatically. |
| **A4** | **Model selects, template phrases — unless the model's sentence passes validation.** | v2 says the model should "phrase a referral sentence"; the digit/`@`/`http` filter is this plan's addition. | Set `REFERRAL_USE_MODEL_PHRASING = False` for a fully templated surface. |
| **A5** | **Coverage = `answer`-routed / 80.** | v2's technical text says exactly this; the Definition of Terms says the looser "answer **or refer/route on** at all", under which coverage is trivially 1.0 and measures nothing. **Recommend tightening the Definition of Terms to match v2's technical wording.** | Report both; only one is informative. |
| **A6** | **PoC selective risk is routing-level only** — a lower bound on the thesis figure. | Answer correctness is human-graded under Objectives 2 and 4, outside Objective 5. | State the bound in Chapter 3; no code change. |
| **A7** | **5-fold grouped cross-validation** of the thresholds. | Not in the notes. Added because fitting thresholds and reporting accuracy on the same 80 queries inflates the headline number. Not a new metric — a better estimator of an existing one. | Drop it and report in-sample accuracy with the caveat stated. |
| **A8** | **Office vocabulary is closed** and `expected_office` must draw from it. | Not specified. Without it, routing accuracy's office breakdown compares strings that can never match. | Relax to fuzzy matching; the breakdown gets noisier. |
| **A9** | **Office table is ~54 rules + 3 document defaults**, not v2's ~25. | Measured: 73 distinct `(doc, root, chapter)` keys, 19 of them bare front-matter pages. | None — this is a measurement, recorded so the effort estimate is honest. |
| **A10** | **Office names in `office_vocabulary` are placeholders.** They are inferred from chapter titles, not verified against a UC-PnC org chart. | The corpus names *sections*, not offices. | **Verify the real office names before Chapter 3.** Substituting correct names later is a data-only edit to `office_lookup.json`; no code changes. |
| **A11** | **`rag_pipeline_office.py` lives on the routing side** so one implementation serves both the build script and the tests. | The repo has no shared package and uses flat imports throughout. | Move it into `rag_pipeline/` and add a path shim; behaviour is identical. |
| **A12** | **Pass 0 / Pass 1 two-pass validation.** Objective 5 is built and unit-tested now against a fake retriever; only Pass 1 numbers are reportable. | Dense retrieval, RRF, and the tokenizer fix are Objective 4's and are not built yet. | If Objective 4 lands first, run Pass 1 directly and skip Pass 0. |

**No contradiction was found between the task description and the Notion notes.** The gate ordering, the three outcomes, the three metrics, the 40/25/15 stratification, the model-assisted referral with deterministic override, and the exemplar-based scope check all match v2 and v3 exactly. The five corpus findings above are gaps between the *notes* and the *code/data*, not disagreements within the design.
