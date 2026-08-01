# Thesis Battle Plan

## Purpose and rule of authority

This is the governing brief for the undergraduate Computer Engineering thesis, **Optimizing Local Small Language Models Using 4-Bit Quantization, QLoRA, and RAG for University Helpdesks**. When a detailed note conflicts with this file, this file takes precedence.

The study is a **controlled prototype evaluation**, not a production deployment, live LMS integration, or post-graduation maintenance commitment.

## Locked study decisions

- One selected locally hosted Small Language Model; the final model name is intentionally not fixed here.
- English university-related queries only.
- Selected student-facing handbooks, rules, policies, and related materials from the University LMS form the local corpus.
- BM25 lexical retrieval is the only retrieval method in the prototype.
- RAG produces answers from retrieved LMS passages with document/page or section citations.
- Lightweight routing returns either a supported RAG answer or a predefined non-answer/office referral. No clarification route is promised.
- QLoRA trains helpdesk behavior, not changeable university facts.
- A 4-bit configuration is compared with a feasible higher-precision reference configuration.
- Base and QLoRA-adapted model versions are compared under the same RAG configuration.
- Evaluation occurs in an 8 GB VRAM controlled test environment.

## Prototype flow

```text
Selected LMS materials -> local knowledge base -> BM25 retrieval
English query -> scope check -> retrieval and evidence check
  -> supported: RAG answer with citation
  -> unsupported/out of scope: predefined non-answer or office referral

Selected SLM -> base and QLoRA-adapted versions -> 4-bit and higher-precision test configurations
All outputs -> quality and performance evaluation
```

## Objectives-to-work mapping

| Objective | Work package | Evidence of completion |
| --- | --- | --- |
| 1. Local 4-bit model | Configure selected SLM on the test environment | Reproducible configuration and resource log |
| 2. Knowledge base | Prepare selected LMS materials and citation metadata | Traceable local corpus |
| 3. BM25 RAG | Retrieve passages and generate cited English answers | Working prototype flow |
| 4. Routing | Handle supported, unsupported, and out-of-scope queries | Documented response paths |
| 5. QLoRA | Train behavior-only adapter and compare with base model | Quality comparison under identical RAG |
| 6. Performance | Compare 4-bit and higher-precision reference | Latency, throughput, and VRAM results |
| 7. Quality evaluation | Apply selected ISO-guided characteristics | Evaluation matrix and results |

## Evaluation plan

Hold the corpus, prompt, query set, decoding settings, and hardware constant whenever comparing configurations.

1. **QLoRA comparison:** base model at 4-bit versus QLoRA-adapted model at 4-bit. Measure answer correctness, groundedness, citation accuracy, instruction following, and unsupported-query handling.
2. **Quantization comparison:** base model at 4-bit versus a feasible higher-precision reference. Measure end-to-end latency, throughput, VRAM consumption, groundedness, and citation accuracy.
3. **Final prototype quality:** assess functional suitability, performance efficiency, response-handling reliability, and source traceability using selected ISO/IEC 25010:2023 and ISO/IEC 25059:2023 characteristics.

The evaluation dataset should include English queries with supporting LMS evidence and English queries without relevant evidence. Record the expected answer or expected referral and the supporting passage for each supported query.

## Scope boundaries

Included: offline prototype, selected student-facing LMS documents available to students, English queries, BM25 RAG, QLoRA behavior adaptation, 4-bit performance testing, and controlled evaluation.

Excluded: live LMS connection, personal student records, multilingual support, dense/hybrid retrieval, reranking, calibrated classifier research, user-satisfaction study, high-concurrency service, production deployment, and ongoing maintenance.

## Execution order

1. Finalize selected LMS corpus and a small labelled English evaluation set.
2. Build the local knowledge base and BM25 retrieval with stable citation metadata.
3. Implement the base-model RAG answer path and non-answer/referral path.
4. Establish base 4-bit and higher-precision performance baselines.
5. Train behavior-only QLoRA and repeat the same evaluation.
6. Organize results under the selected ISO quality characteristics.
7. Update the manuscript only with findings supported by the final measurements.

## Risks to manage

- LMS material may be incomplete, outdated, or inconsistent; record source dates and limitations.
- BM25 may miss synonyms or paraphrases; report this as a retrieval limitation rather than adding dense retrieval without approval.
- A full-precision configuration may not fit the test environment; use a feasible higher-precision reference and document it.
- RAG and citations reduce unsupported answers but do not guarantee correctness; preserve the non-answer/referral path.

## Document map

- [README.md](README.md): concise project context and document index.
- [ArchitectureAndDesign.md](ArchitectureAndDesign.md): system components, data flow, and experiments.
- [prompt_and_routing_architecture.md](prompt_and_routing_architecture.md): prompt contract, routing, output, and citation validation.
- [ISO_STANDARDS_AND_THESIS_CONTEXT.md](ISO_STANDARDS_AND_THESIS_CONTEXT.md): ISO-guided evaluation mapping and thesis boundaries.
