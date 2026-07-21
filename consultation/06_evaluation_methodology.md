# 06. Evaluation Methodology

## Goal

Measure the system rigorously enough to defend claims about retrieval quality, grounding, abstention, multilingual behavior, latency, and memory. Retrieval and generation are evaluated separately, then combined.

## Two evaluation sets

### Retrieval set

Each item includes:

- question text;
- relevant document IDs;
- relevant pages or sections;
- acceptable supporting chunks;
- language;
- query category;
- difficulty;
- whether an answer exists.

### End-to-end answer set

Include:

- answerable questions;
- unanswerable questions;
- ambiguous questions;
- conflicting-document questions;
- outdated-policy questions;
- out-of-scope questions;
- adversarial and prompt-injection questions;
- privacy-sensitive requests;
- multilingual and code-switched questions.

## Language strata

| Group | Example |
|---|---|
| English | Fully English question |
| Filipino | Fully Filipino question |
| Code-switched | Mixed Filipino-English |
| Institutional terms | Acronyms, course codes, office names |

Bilingual evaluators should assess meaning preservation, because citation correctness can stay high while a Filipino paraphrase distorts meaning.

## Retrieval metrics

- Recall@k;
- MRR;
- nDCG;
- retrieval latency at target settings;
- language-stratified results.

## Evidence-sufficiency metrics

- precision, recall, F1;
- calibration error;
- coverage and selective risk at the chosen threshold.

## Answer metrics

- correctness;
- groundedness;
- citation presence, validity, entailment, completeness.

## Selective-answering metrics

$$
\operatorname{Coverage}=\frac{\text{answered}}{\text{all questions}}
$$

$$
\operatorname{SelectiveRisk}=\frac{\text{incorrect answered}}{\text{answered}}
$$

A system that abstains on everything has near-zero hallucination but no utility. Always report coverage with error rate.

## Outcome taxonomy

Do not fold clarification into either success or failure. Track:

- correct answer;
- correct abstention;
- useful clarification;
- unnecessary clarification;
- unsupported answer;
- wrong refusal.

## Failure-stage attribution

Label where an end-to-end failure originated:

1. retrieval;
2. reranking;
3. evidence-sufficiency;
4. generation;
5. citation mapping.

This is one of the most valuable analyses in the thesis.

## Human evaluation

- define rubric for correctness, grounding, and safety;
- use multiple annotators on a subset;
- report inter-annotator agreement;
- keep annotation guidelines with the thesis.

## Statistical rigor

- run key training configurations at least three times where feasible;
- report confidence intervals for human ratings;
- report bootstrap intervals for retrieval metrics;
- label single-run exploratory experiments.

## Controlled comparison rules

- replay identical retrieved evidence to Q0, Q1, and Q2;
- hold prompt, decoding, quantization, and evidence constant across variants;
- change one factor per ablation.

## Reporting template per configuration

| Field | Value |
|---|---|
| Variant | Q0 / Q1 / Q2 |
| Prompt | P0 / P1 / P2 / P3 |
| Quantization | e.g. Q4_K_M |
| Context length | tokens |
| Recall@k, MRR, nDCG | |
| Sufficiency P/R/F1 | |
| Coverage, selective risk | |
| Groundedness, citation metrics | |
| Latency, TTFT, throughput | |
| RAM, VRAM, concurrency | |

## Common pitfalls

- Reporting one accuracy number without coverage.
- Random splits leaking paraphrases into the test set.
- Ignoring multilingual meaning preservation.
- Single-run results presented as stable.
- No failure-stage attribution.

## References

- Geifman and El-Yaniv, “Selective Classification for Deep Neural Networks,” 2017: https://arxiv.org/abs/1705.08500
- Guo et al., “On Calibration of Modern Neural Networks,” 2017: https://proceedings.mlr.press/v70/guo17a.html
- Järvelin and Kekäläinen, “Cumulated Gain-Based Evaluation of IR Techniques (nDCG),” 2002: https://doi.org/10.1145/582415.582418
- Es et al., “RAGAS: Automated Evaluation of Retrieval Augmented Generation,” 2023: https://arxiv.org/abs/2309.15217
