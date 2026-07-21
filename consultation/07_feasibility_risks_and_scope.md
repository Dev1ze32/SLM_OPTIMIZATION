# 07. Feasibility, Risks, and Scope

## Purpose

Define what is realistically achievable, what is a stretch goal, and what is likely infeasible as originally described. This protects the thesis from overpromising.

## Feasible core

A single-node, intranet or offline, evidence-based university helpdesk using a 3B quantized model on a machine planning around 8GB VRAM, serving a controlled corpus of approved documents, with retrieval, evidence-aware routing, behavior QLoRA, and separate retrieval and end-to-end evaluation.

## Stretch goals requiring measurement

- High concurrency on one 8GB GPU.
- Strong Filipino and code-switched generation quality.
- Very long effective context.
- Simultaneous GPU residency for generator, BGE-M3, and reranker.
- Fully automated citation-entailment scoring.

## Likely infeasible as described

| Item | Problem | Mitigation |
|---|---|---|
| All models resident on 8GB GPU at once | Memory pressure | Reserve GPU for generation; CPU or selective loading for retrieval models |
| High concurrency without queueing | Single-GPU throughput limits | Add a request queue; report concurrency limits |
| Guaranteed fluent Filipino generation | Filipino is not a prominently supported Llama 3.2 language | Cited English fallback when confidence is low |
| Long context as default | Latency and memory cost | Short effective context; expand only if evaluation requires |
| Easy fully air-gapped maintenance | Operational overhead | Document transfer, update, and audit procedures |

## Multilingual risk

Retrieval may find the correct English policy while the generator distorts it in Filipino. Citation correctness can remain high while meaning degrades. A defensible production policy:

- accept multilingual queries;
- retrieve institutional sources regardless of language;
- answer in the query language only when confidence is adequate;
- otherwise give a concise cited English answer.

## Training versus deployment hardware

Training may use a larger temporary GPU; deployment is the constrained machine. This does not weaken offline claims as long as the two environments are clearly separated. QLoRA training needs memory for activations, gradients, optimizer state, and the adapter, so 8GB training is not implied by 8GB inference.

## Air-gapped operations

Plan for:

- transferring models and dependencies in;
- dependency installation and vulnerability updates;
- document update and re-index procedures;
- model-license records;
- backups and audit logs;
- administrator authentication;
- index rebuild process.

## Privacy and security

- Personal student records stay outside the public knowledge-base path.
- Access filters run before evidence reaches the generator.
- Documents are untrusted data; embedded instructions never override policy.
- Minimize personal content in logs.
- Ingestion and approval require authentication and auditability.

## Scope boundaries for the thesis

In scope:

- approved institutional information;
- grounding, citation, abstention, clarification;
- retrieval and quantization trade-offs;
- Q0/Q1/Q2 comparison;
- constrained-hardware benchmarking.

Out of scope unless explicitly extended:

- transactional actions such as enrolling or paying;
- authenticated per-student record access;
- open-domain conversation;
- guaranteed production SLA.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Table extraction failure | High | High | Table-aware normalization |
| Over-abstention | Medium | Medium | Varied abstention data; calibrated threshold |
| Confident unsupported answers | Medium | High | Sufficiency calibration; citation validation |
| Filipino quality shortfall | Medium | Medium | Cited English fallback |
| 8GB memory overrun | Medium | High | CPU retrieval; benchmarking |
| Evaluation leakage | Medium | High | Split by document, template, paraphrase |
| Runtime nondeterminism | Low | Medium | Pin versions; record commands |

## References

- Meta Llama 3.2 model card and supported languages: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Meta Llama 3.2 license and use policy: https://www.llama.com/llama3_2/license/
- Dettmers et al., “QLoRA,” 2023: https://arxiv.org/abs/2305.14314
- llama.cpp: https://github.com/ggml-org/llama.cpp
