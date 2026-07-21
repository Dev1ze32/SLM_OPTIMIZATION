# 01. AI Project Context

## How to use this file

This document is the canonical handoff for another AI assistant or researcher. It should be readable without the original consultation chat. Read this file first, then follow the document map.

## Thesis in one paragraph

The project studies whether a small, locally deployed, quantized language model can serve as a reliable, evidence-grounded university helpdesk under constrained hardware and offline or intranet conditions. It combines an approved institutional corpus, hybrid retrieval, evidence-aware routing, behavior-oriented QLoRA, and quantized inference, and it measures the resulting quality, latency, memory, and reliability trade-offs.

## What is being built

- An offline-capable RAG system for university information.
- A retrieval pipeline over approved institutional documents.
- An evidence-aware router that answers, clarifies, or abstains.
- A QLoRA-tuned small instruction model for behavior, not fact storage.
- A quantized deployment for constrained hardware.
- An evaluation framework separating retrieval from generation.

## What is explicitly not the goal

- A general-purpose assistant.
- A model that memorizes changeable institutional facts.
- A production system with proven high concurrency from the outset.
- A guaranteed-fluent multilingual generator.

## Target hardware assumptions

- Deployment: a single constrained campus machine, planning around 8GB VRAM.
- Network: offline or intranet-only.
- Training: may occur on a larger temporary GPU; this does not weaken offline deployment claims if training and inference environments are clearly distinguished.

Any claim that a configuration fits 8GB must be demonstrated with benchmarks, not inferred from model-file size.

## Fixed terminology

| Term | Meaning |
|---|---|
| Q0 | Untuned instruction model plus identical RAG |
| Q1 | Behavior-only QLoRA plus identical RAG |
| Q2 | Behavior-plus-facts QLoRA plus identical RAG |
| Behavior | Grounding, citation, abstention, brevity, safety, tone |
| Facts | Changeable institutional data such as dates, fees, contacts |
| Evidence sufficiency | Calibrated estimate that retrieved evidence can support an answer |
| Citation validity | Identifier maps to evidence actually retrieved |
| Citation entailment | Cited passage supports the claim |
| Selective risk | Error rate among answered questions |
| Coverage | Fraction of questions answered rather than deferred |

## Current design decisions

- Keep changeable facts in retrieval; use QLoRA for stable behavior.
- Do not equate low retrieval score with chit-chat.
- Use calibrated evidence sufficiency, not raw RRF thresholds.
- Assign citation identifiers server-side and validate them.
- Prefer at most one generation call for a normal grounded answer.
- Preserve table structure and document authority metadata.
- Prioritize GPU memory for generation; benchmark embeddings and reranking on CPU.

## Corrected assumptions from the original plan

| Original assumption | Correction |
|---|---|
| Low retrieval score implies casual conversation | Retrieve, then assess evidence sufficiency; misclassifying a real question is a critical error |
| RRF score can be thresholded as confidence | Calibrate sufficiency on labeled data with reported precision, recall, and calibration |
| Citation string proves grounding | Separate presence, validity, entailment, and completeness |
| Quantization necessarily degrades long prompts | Treat prompt effects as a testable hypothesis on the final artifact |
| 8GB VRAM is automatically enough | Distinguish training from inference; benchmark concurrency and residency |
| Behavior-plus-facts tuning is obviously best | Include an untuned baseline; treat Q2 as an ablation |

## Open research questions

- Which chunking strategy best preserves tables, schedules, and directories?
- Which embedding and reranking models balance multilingual quality with CPU latency?
- What evidence-sufficiency threshold optimizes the coverage-risk trade-off?
- Does QLoRA improve behavior over a well-prompted untuned baseline?
- How much does Q2 suffer when policies change after training?
- What is the acceptable concurrency ceiling before queueing is required?
- Is Filipino generation quality acceptable, or should answers fall back to cited English?

## Evidence-labeling convention

Every consequential statement in this package is one of:

- **Established guidance** verified from official documentation or a primary paper.
- **Project recommendation** that is reasonable but institution-specific.
- **Hypothesis** requiring evaluation before acceptance.

## Document map

- `README.md` — top-level thesis plan.
- `prompt_and_routing_architecture.md` — routing, prompting, and citation policy.
- `02_architecture_review.md` — architecture assessment and revised design.
- `03_qlora_fine_tuning.md` — fine-tuning experiments and configuration.
- `04_retrieval_and_knowledge_base.md` — ingestion, chunking, retrieval, metadata.
- `05_constrained_inference_optimization.md` — quantized deployment and benchmarking.
- `06_evaluation_methodology.md` — test sets and metrics.
- `07_feasibility_risks_and_scope.md` — feasibility and scope boundaries.
- `08_execution_roadmap.md` — ordered phases and decision gates.

## Guidance for a continuing AI

- Do not reintroduce corrected assumptions.
- Keep Q0/Q1/Q2 terminology consistent.
- Do not claim hallucinations are eliminated or that temperature zero guarantees correctness.
- Prefer measured results over intuition for any hardware or latency claim.
- When adding runtime flags or library APIs, verify against the pinned versions rather than memory.

## References

- Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” 2020: https://arxiv.org/abs/2005.11401
- Dettmers et al., “QLoRA,” 2023: https://arxiv.org/abs/2305.14314
- Meta Llama 3.2 3B Instruct model card: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
