# ISO Standards and Thesis Context

## Purpose

This note maps the prototype evaluation to relevant quality standards. The standards are used as evaluation references only; the thesis does not claim ISO compliance or certification.

## Study context

The study develops and evaluates an English-only, offline university-helpdesk prototype. It uses selected student-facing LMS materials, hybrid retrieval (BM25 + dense embedding with RRF fusion), RAG, a selected local SLM, QLoRA behavior adaptation, and 4-bit performance testing in an 8 GB VRAM environment.

The governing decisions are in [THESIS_BATTLE_PLAN.md](THESIS_BATTLE_PLAN.md).

## Standards used

### ISO/IEC 25010:2023 - Product quality model

Use selected characteristics to organize prototype evaluation:

- **Functional suitability:** answer correctness, groundedness, citation accuracy, and citation validity (the connection between a response and its retrieved LMS passages — the identifier a response cites must resolve to evidence actually supplied to the model).
- **Performance efficiency:** end-to-end latency, throughput, and VRAM consumption.
- **Reliability of response handling:** consistent supported-answer versus non-answer/referral handling under the defined test cases.

## Evaluation matrix

This table organizes measurements defined in [EVALUATION_PLAN.md](EVALUATION_PLAN.md); it adds no measurement of its own.

| Quality focus | Prototype measure | Scoring | Evidence source |
| --- | --- | --- | --- |
| Functional suitability | correctness, groundedness, citation entailment | human, rubric-scored | answerable test cases with gold chunks |
| Functional suitability (traceability) | citation validity (identifier resolves to supplied evidence) | automatic | local knowledge-base metadata |
| Performance efficiency | latency, throughput, peak VRAM | automatic, ≥3 runs | controlled test-environment logs |
| Response-handling reliability | correct answer or referral route; coverage and selective risk | automatic | all three query strata |

## Required wording in the manuscript

Use this statement consistently:

> The prototype evaluation is guided by selected measurable quality characteristics from ISO/IEC 25010:2023. This standard is used as an evaluation reference only and does not indicate formal ISO certification.

## References

- ISO/IEC 25010:2023, *Systems and software engineering - Systems and software Quality Requirements and Evaluation (SQuaRE) - Product quality model*, 2nd ed., Nov. 2023. https://www.iso.org/standard/78176.html

