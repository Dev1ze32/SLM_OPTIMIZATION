# Offline University Helpdesk Prototype

## What this project is

This thesis develops and evaluates an offline-capable **prototype** university helpdesk. It uses a selected locally hosted Small Language Model, BM25-based Retrieval-Augmented Generation, QLoRA behavior adaptation, and 4-bit quantization to provide citation-backed responses to English university-related queries.

The local knowledge base is built from selected student-facing handbooks, rules, policies, and related materials from the University Learning Management System.

## What this project is not

It is not a production deployment, a live LMS integration, a multilingual assistant, a dense/hybrid retrieval study, or a system that processes personal student records. It creates no post-study maintenance obligation.

## Canonical decisions

- One selected SLM; do not name a final model until selection is complete.
- English queries only.
- BM25 only; no dense retrieval, RRF, or reranking.
- Two routing outcomes: supported RAG answer, or predefined non-answer/office referral.
- QLoRA trains response behavior, not university facts.
- Compare base versus QLoRA-adapted behavior at the same 4-bit configuration.
- Compare 4-bit with a feasible higher-precision reference configuration.
- Test in a controlled 8 GB VRAM environment.

## Read this first

[THESIS_BATTLE_PLAN.md](THESIS_BATTLE_PLAN.md) is the governing high-level plan. It overrides conflicting details in all other notes.

## Detailed documents

- [Architecture and Design](ArchitectureAndDesign.md)
- [Prompt and Routing Architecture](prompt_and_routing_architecture.md)
- [ISO Standards and Thesis Context](ISO_STANDARDS_AND_THESIS_CONTEXT.md)

