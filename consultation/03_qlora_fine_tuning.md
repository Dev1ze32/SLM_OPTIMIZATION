# 03. QLoRA Fine-Tuning

## Objective

Determine whether QLoRA improves grounded helpdesk behavior, and whether including changeable facts in training harms reliability when policies change. Behavior means grounding, citation, abstention, brevity, safety, and tone. Facts means changeable institutional data.

## Required experimental variants

| ID | Model | Role |
|---|---|---|
| Q0 | Untuned instruction model + identical RAG | Baseline; determines if tuning is needed |
| Q1 | Behavior-only QLoRA + identical RAG | Learned behavior |
| Q2 | Behavior-plus-facts QLoRA + identical RAG | Stale-memory and conflict ablation |

Without Q0 you cannot claim QLoRA helped. A well-prompted untuned instruction model may match a tuned adapter.

### Controlled comparison

Hold everything except the adapter constant: identical retrieval results, prompt template, decoding parameters, evidence, quantization, and evaluation questions. For a clean generation comparison, cache retrieved evidence per test question and replay identical evidence to all variants.

## Training quantization configuration

Established guidance for a 4-bit QLoRA setup:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # fall back to float16 if unsupported
)
```

Enable gradient checkpointing. Confirm dtype support on the actual training GPU before assuming BF16.

## LoRA module targets

Compare two configurations rather than assuming one:

- Lower cost: `q_proj, k_proj, v_proj, o_proj`.
- Higher capacity: all linear layers.

Because target behavior is narrow, attention projections may already be sufficient. All-linear increases trainable parameters and memory use.

## Bounded hyperparameter candidates

| Parameter | Candidates |
|---|---|
| Rank | 8, 16, 32 |
| Alpha | 16, 32, 64 |
| Dropout | 0.0 or 0.05 |
| Learning rate | 1e-4, 2e-4 |
| Epochs | 1 to 3 |
| Sequence length | 512 or 1024 |
| Effective batch size | 8 to 32 via accumulation |
| Scheduler | cosine or linear |
| Warmup ratio | 0.03 to 0.1 |
| Optimizer | paged 8-bit AdamW |

Keep the grid small. A rank comparison of 8 versus 16 versus 32 is enough unless optimization is itself the thesis topic.

## Data format

Use the checkpoint's official chat template and compute loss on assistant output only. The model should not spend capacity reproducing system instructions or user questions.

```json
{
  "messages": [
    {"role": "system", "content": "Short production system instruction"},
    {"role": "user", "content": "When is late enrollment?"},
    {"role": "assistant", "content": "I don't have enough evidence to confirm the late-enrollment date. Please check with the Registrar's Office."}
  ]
}
```

Exact loss-masking and packing implementation depend on the installed TRL version; pin versions and document them.

## Behavior dataset categories

- single-source supported answers;
- multi-source supported answers;
- conflicting documents;
- outdated versus current documents;
- insufficient evidence;
- ambiguous questions;
- follow-ups with missing context;
- restricted personal-data requests;
- prompt-injection text inside documents;
- out-of-scope questions;
- English, Filipino, and code-switched queries;
- malformed or missing citations;
- concise versus overly broad questions.

Vary abstention wording and situation to avoid teaching the model to over-abstain.

## Leakage prevention

Split by source document, intent template, and paraphrase family, not by random row. Paraphrases of a training question must not appear in the test set. Otherwise the evaluation measures memorization.

## Stale-fact experiment for Q2

1. Train Q2 on policy version A.
2. Replace the retrieval corpus with conflicting policy version B.
3. Ask questions where A and B disagree.
4. Measure whether the model follows current evidence B or recalls stale fact A.

This is one of the most compelling results the thesis can produce.

## Sequence packing

Packing improves efficiency for short samples, but verify masking and conversation boundaries so one training conversation does not leak into another through attention or loss masks.

## Deployment order

```text
base model
  -> train LoRA adapter
  -> evaluate adapter (pre-merge)
  -> merge adapter into a compatible base checkpoint
  -> convert merged model to GGUF
  -> quantize GGUF
  -> evaluate quantized artifact again
```

Adapter behavior in Transformers does not guarantee identical behavior after merging and GGUF quantization; re-evaluate the final artifact.

## Reproducibility requirements

- Pin versions of Transformers, PEFT, TRL, bitsandbytes, and the conversion tool.
- Record seeds and, where feasible, run key configurations at least three times.
- Log dataset version, split method, and template.
- Save adapter, merged, and quantized artifacts with hashes.
- Report single-run exploratory experiments explicitly.

## Common pitfalls

- No untuned baseline.
- Loss over the full prompt instead of assistant output.
- Random splits causing leakage.
- Identical abstention text causing over-refusal.
- Evaluating only the pre-merge adapter.
- Treating Q2 as a foregone loser instead of a measured ablation.

## References

- Dettmers et al., “QLoRA,” 2023: https://arxiv.org/abs/2305.14314
- Hu et al., “LoRA,” 2021: https://arxiv.org/abs/2106.09685
- Hugging Face PEFT documentation: https://huggingface.co/docs/peft/
- Hugging Face TRL SFT documentation: https://huggingface.co/docs/trl/sft_trainer
- bitsandbytes quantization guide: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- llama.cpp conversion and quantization: https://github.com/ggml-org/llama.cpp
