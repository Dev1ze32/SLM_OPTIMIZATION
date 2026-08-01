# ISO Standards and Thesis Context

## Purpose

This note maps the prototype evaluation to relevant quality standards. The standards are used as evaluation references only; the thesis does not claim ISO compliance or certification.

## Study context

The study develops and evaluates an English-only, offline university-helpdesk prototype. It uses selected student-facing LMS materials, BM25 retrieval, RAG, a selected local SLM, QLoRA behavior adaptation, and 4-bit performance testing in an 8 GB VRAM environment.

The governing decisions are in [THESIS_BATTLE_PLAN.md](THESIS_BATTLE_PLAN.md).

## Standards used

### ISO/IEC 25010:2023 - Product quality model

Use selected characteristics to organize prototype evaluation:

- **Functional suitability:** answer correctness, groundedness, and citation accuracy.
- **Performance efficiency:** end-to-end latency, throughput, and VRAM consumption.
- **Reliability of response handling:** consistent supported-answer versus non-answer/referral handling under the defined test cases.

### ISO/IEC 25059:2023 - Quality model for AI systems

Use this as the AI-system quality context for source traceability and evidence-grounded response behavior. The relevant prototype evidence is the connection between a response and its retrieved LMS passages.

## Evaluation matrix

| Quality focus | Prototype measure | Evidence source |
| --- | --- | --- |
| Functional suitability | correctness, groundedness, citation accuracy | labelled English test cases and retrieved passages |
| Performance efficiency | latency, throughput, peak VRAM | controlled test-environment logs |
| Response-handling reliability | correct answer or referral route | supported and unsupported test cases |
| Source traceability | valid document title and page/section citation | local knowledge-base metadata |

## Required wording in the manuscript

Use this statement consistently:

> The prototype evaluation is guided by selected measurable quality characteristics from ISO/IEC 25010:2023 and ISO/IEC 25059:2023. These standards are used as evaluation references only and do not indicate formal ISO certification.

## References

- ISO/IEC 25010:2023, *Systems and software engineering - Systems and software Quality Requirements and Evaluation (SQuaRE) - Product quality model*, 2nd ed., Nov. 2023. https://www.iso.org/standard/78176.html
- ISO/IEC 25059:2023, *Software engineering - Systems and software Quality Requirements and Evaluation (SQuaRE) - Quality model for AI systems*, 1st ed., Jun. 2023. https://www.iso.org/standard/80655.html

