# 05. Constrained Inference Optimization

## Goal

Deploy the selected model on constrained campus hardware while measuring quality, latency, memory, and concurrency. All optimization must be evidence-driven, not assumed.

## Principle: measure before tuning

Establish a baseline before changing any setting. Record:

- model file size;
- idle and peak VRAM;
- idle and peak RAM;
- prompt-processing throughput;
- generation throughput;
- time to first token;
- total latency;
- output quality;
- effective context length;
- concurrent request count.

A configuration is only "fits 8GB" once benchmarked. Model-file size alone does not prove it, because KV cache, runtime overhead, and concurrent models also consume memory.

## Memory placement

Prioritize GPU memory for generation. Do not assume the generator, BGE-M3, and a cross-encoder reranker can all reside on an 8GB GPU simultaneously. Benchmark embeddings and reranking on CPU, or load them selectively.

## Context policy

A 128K theoretical context window is not a useful deployment target here. Start with a short effective context, roughly 2K to 4K tokens, and increase only if evaluation shows missing context.

Compose the prompt from:

- a short system instruction;
- top-three evidence chunks;
- only necessary conversation history;
- a strict output-token limit;
- parent context only when needed.

## Quantization comparison

Do not evaluate a single 4-bit artifact. Compare across the quality-memory frontier:

| Format | Purpose |
|---|---|
| Highest feasible precision | Reference quality |
| Q8 | High-quality quantized baseline |
| Q5_K_M | Middle point |
| Q4_K_M | Main constrained candidate |
| Q3 (optional) | Stress test |

If the reference cannot run on the target machine, evaluate it on separate hardware and report that clearly.

## Decoding

Begin conservative for helpdesk answers:

- temperature approximately 0 to 0.3;
- limited maximum output tokens;
- deterministic seed where supported;
- no creative sampling by default.

Temperature zero reduces sampling variability; it does not guarantee correctness or prevent hallucination.

## Concurrency testing

A model fitting in VRAM does not prove campus-scale concurrency. Measure:

| Load | Metrics |
|---|---|
| 1 request | TTFT, throughput, latency |
| 2 concurrent | Added latency, peak memory |
| 4 concurrent | Queue wait, stability |

A single 8GB GPU may realistically require a request queue. That is an acceptable constrained-system design if reported honestly.

## Runtime reproducibility

llama.cpp changes quickly. Pin the release or commit and record the exact command, model artifact, GPU driver, and backend. Benchmark features such as flash attention and KV-cache quantization on the actual hardware rather than assuming they always help. Do not present bleeding-edge flags as universal recommendations.

## Benchmark matrix

Vary one factor at a time:

- quantization level;
- context length;
- number of evidence chunks;
- reranker on or off;
- concurrency level;
- prompt variant P0/P1/P2/P3.

Record every metric per cell so trade-offs are visible.

## Optimization order

1. Establish measured baseline.
2. Shorten context and cap output.
3. Compare quantization levels.
4. Fix conservative decoding.
5. Test concurrency and add queueing if needed.
6. Only then test optional runtime features, each benchmarked.

## Common pitfalls

- Inferring memory fit from file size.
- Running all models on the GPU at once.
- Targeting a very long context without need.
- Assuming a runtime flag improves speed or quality.
- Reporting single-request latency as production capacity.

## References

- llama.cpp: https://github.com/ggml-org/llama.cpp
- llama.cpp quantization overview: https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md
- Meta Llama 3.2 3B Instruct model card: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Dettmers et al., “QLoRA,” 2023: https://arxiv.org/abs/2305.14314
