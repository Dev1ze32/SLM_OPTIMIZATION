# Thesis Battle Plan

## Purpose and rule of authority

This is the governing brief for the undergraduate Computer Engineering thesis, **Optimizing Local Small Language Models Using 4-Bit Quantization, QLoRA, and RAG for University Helpdesks**. When a detailed note conflicts with this file, this file takes precedence.

The study is a **controlled prototype evaluation**, not a production deployment, live LMS integration, or post-graduation maintenance commitment.

## Locked study decisions

- One selected locally hosted Small Language Model; the final model name is intentionally not fixed here.
- English university-related queries only.
- Selected student-facing handbooks, rules, policies, and related materials from the University LMS form the local corpus.
- Hybrid retrieval (BM25 lexical + dense embedding) retrieves passages; RRF fusion reranks candidates; disjunctive sufficiency gate admits answers when either retriever exceeds its threshold.
- Dense retrieval uses exhaustive cosine similarity over an in-memory embedding array, **not** a vector database. At 364 chunks (~550 KB of vectors) exhaustive search returns *exact* nearest neighbors in under a millisecond; an approximate-nearest-neighbor index would add infrastructure and approximation error with no retrieval benefit at this scale. Embeddings are cached to disk so they are computed once, not per run.
- RAG produces answers from retrieved LMS passages with document/page or section citations.
- Lightweight routing returns either a supported RAG answer or a predefined non-answer/office referral. No clarification route is promised.
- QLoRA trains helpdesk behavior, not changeable university facts.
- A 4-bit configuration is compared with an 8-bit reference configuration — the feasible higher-precision point within the 8 GB VRAM budget; full 16-bit does not fit.
- Base and QLoRA-adapted model versions are compared under the same RAG configuration.
- Evaluation occurs in an 8 GB VRAM controlled test environment.

## Prototype flow

```text
Selected LMS materials -> local knowledge base -> BM25 retrieval
English query -> retrieval and evidence check -> scope check only if insufficient
  -> supported: RAG answer with citation
  -> unsupported/out of scope: predefined non-answer or office referral

Selected SLM -> base and QLoRA-adapted versions -> 4-bit and 8-bit test configurations
All outputs -> quality and performance evaluation
```

## Objectives-to-work mapping

| Objective | Work package | Evidence of completion |
| --- | --- | --- |
| 1. Model configuration and adaptation | Configure the selected SLM in 4-bit (and 8-bit reference) precision on the test environment; train and apply the behavior-only QLoRA adapter | Reproducible configuration/resource log; peak VRAM, latency, throughput (4-bit vs. 8-bit); answer correctness, groundedness, citation accuracy (base vs. QLoRA) |
| 2. Knowledge base | Prepare selected LMS materials and citation metadata | Traceable local corpus; metadata completeness and citation validity rates |
| 3. Hybrid retrieval RAG | Retrieve passages (BM25 + dense embedding, RRF fusion) and generate cited English answers | Working prototype flow; retrieval precision (Recall@k/MRR), answer accuracy, and error rate for the hybrid configuration |
| 4. Routing | Handle supported, unsupported, and out-of-scope queries | Documented response paths; routing accuracy, coverage, selective risk |
| 5. ISO-guided quality evaluation | Apply selected ISO/IEC 25010:2023 characteristics to the measurements from objectives 1, 3, and 4 | Evaluation matrix mapping functional suitability, performance efficiency, and reliability to existing results |

Objective 5 adds no new measurement — it maps numbers already produced by
objectives 1, 3, and 4 onto ISO/IEC 25010:2023 characteristics. See
[ISO_STANDARDS_AND_THESIS_CONTEXT.md](ISO_STANDARDS_AND_THESIS_CONTEXT.md).

## Evaluation plan

Hold the corpus, prompt, query set, decoding settings, and hardware constant whenever comparing configurations. Comparisons are one-factor-at-a-time from a shared anchor, never factorial.

Three configurations, all at hybrid retrieval:

| ID | Model | Precision | Role |
| --- | --- | --- | --- |
| A | base | 4-bit | Anchor; shared by both comparisons |
| B | QLoRA-adapted | 4-bit | A vs B isolates the adapter |
| C | base | 8-bit | A vs C isolates quantization |

1. **QLoRA comparison (A vs B):** measure answer correctness, groundedness, citation accuracy, instruction following, and unsupported-query handling.
2. **Quantization comparison (A vs C):** measure end-to-end latency, throughput, VRAM consumption, groundedness, and citation accuracy.
3. **Hybrid RAG pipeline evaluation:** measure retrieval precision (Recall@k/MRR), answer accuracy, and error rate for the hybrid configuration. Retrieval stays fixed at hybrid for A, B, and C; a BM25-only/dense-only comparison is out of scope for this controlled evaluation and is not run — hybrid retrieval's design is justified by corpus-specific reasoning instead (see Locked study decisions), and listed as future work.
4. **Final prototype quality:** assess functional suitability, performance efficiency, and reliability using selected ISO/IEC 25010:2023 characteristics.

The evaluation dataset is roughly 80 stratified English queries: answerable from the corpus, in scope without evidence, and out of scope. Record the expected answer or expected referral for each, and gold supporting chunks for the answerable ones.

Retrieve once per query and replay identical cached evidence to A, B, and C so that observed differences are attributable to the varied factor rather than to retrieval variance.

[EVALUATION_PLAN.md](EVALUATION_PLAN.md) is the detailed measurement design: query-set composition, the pooling protocol for gold-chunk annotation, the automatic-versus-human metric split, run counts, effort budget, and the triage order if the schedule collapses.

## Scope boundaries

Included: offline prototype, selected student-facing LMS documents available to students, English queries, hybrid retrieval RAG (BM25 + dense embedding with RRF fusion), QLoRA behavior adaptation, 4-bit performance testing, and controlled evaluation.

Excluded: live LMS connection, personal student records, multilingual support, reranking beyond RRF, calibrated classifier research (Gate 1 remains an exemplar-based nearest-neighbor lookup), vector-database or approximate-nearest-neighbor infrastructure, user-satisfaction study, high-concurrency service, production deployment, and ongoing maintenance.

## Execution order

1. Finalize selected LMS corpus and a small labelled English evaluation set.
2. Build the local knowledge base and BM25 retrieval with stable citation metadata.
3. Implement the base-model RAG answer path and non-answer/referral path.
4. Establish base 4-bit and 8-bit performance baselines.
5. Train behavior-only QLoRA and repeat the same evaluation.
6. Organize results under the selected ISO quality characteristics.
7. Update the manuscript only with findings supported by the final measurements.

## Risks to manage

- LMS material may be incomplete, outdated, or inconsistent; record source dates and limitations.
- Hybrid retrieval (BM25 handles keyword/exact-token queries; dense embedding handles paraphrase/intent) reduces the likelihood either retriever alone misses relevant passage. The disjunctive sufficiency gate admits answers when either retriever exceeds its component threshold, not when both fail.
- A full 16-bit configuration does not fit the 8 GB VRAM test environment: for a ~3-4B parameter model, fp16 weights alone require roughly 6-8 GB, leaving no headroom for KV-cache, activations, or framework overhead. An 8-bit configuration is used instead as the feasible higher-precision reference; this substitution is documented as a hardware-driven decision, not an oversight.
- RAG and citations reduce unsupported answers but do not guarantee correctness; preserve the non-answer/referral path.
- Hybrid retrieval is not empirically compared against BM25-only or dense-only retrieval in this study; its design is justified by corpus-specific reasoning (form codes/acronyms favor lexical matching, paraphrased student queries favor semantic matching) and cited literature instead. State this explicitly as a limitation and list the empirical ablation as future work.

## Document map

- [README.md](README.md): concise project context and document index.
- [ArchitectureAndDesign.md](ArchitectureAndDesign.md): system components, data flow, and experiments.
- [prompt_and_routing_architecture.md](prompt_and_routing_architecture.md): prompt contract, routing, output, and citation validation.
- [EVALUATION_PLAN.md](EVALUATION_PLAN.md): measurement design, query set, annotation protocol, effort budget, triage order.
- [ISO_STANDARDS_AND_THESIS_CONTEXT.md](ISO_STANDARDS_AND_THESIS_CONTEXT.md): ISO-guided evaluation mapping and thesis boundaries.
- [docs/TABLE_HANDLING_PLAN.md](docs/TABLE_HANDLING_PLAN.md): PDF table extraction rules (grid vs. outline classification, indent recovery) and the ambiguous-case flag report.

`consultation/` contains superseded design notes from an earlier, larger scope (clarification route, multilingual evaluation, reranking, Q2 adapter, concurrency testing). Retained for design rationale only. Do not cite it in the manuscript or reintroduce its scope.
