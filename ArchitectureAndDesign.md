# Constrained University Helpdesk System Design

## 1. Purpose

This document defines the proposed architecture for an offline-capable, retrieval-augmented university helpdesk deployed on hardware with an **8 GB VRAM GPU**. The design prioritizes:

- accurate, evidence-grounded answers;
- precise responses with citations;
- low and predictable latency;
- conservative behavior when evidence is weak;
- feasible CPU, RAM, and VRAM consumption;
- operation without depending on cloud inference.

The central design rule is to reserve the GPU primarily for **one quantized answer-generation model**. Retrieval, reranking, policy checks, and evidence routing run on the CPU.

---

## 2. Hardware Assumptions

The initial target environment is:

- GPU with 8 GB VRAM;
- conventional multi-core CPU;
- sufficient system RAM for document indexes and lightweight CPU models;
- local disk storage;
- limited or unavailable internet access during normal operation.

Exact CPU and RAM requirements will be determined through benchmarking on the target campus computer.

---

## 3. High-Level Architecture

```text
Approved university documents
          |
          v
Offline ingestion and indexing
  - extraction and cleaning
  - chunking
  - metadata assignment
  - BM25 index
  - dense vector index
          |
          v
User question
          |
          v
1. Policy and access checks                [CPU, rules]
          |
          v
2. Intent and query checks                 [CPU, rules/classifier]
          |
          v
3. BM25 + dense retrieval                  [CPU]
          |
          v
4. Reciprocal Rank Fusion                  [CPU]
          |
          v
5. Conditional reranking                   [CPU, optional]
          |
          v
6. Evidence-sufficiency routing            [CPU]
       /          |           \
      /           |            \
  Answer       Clarify       Abstain
     |             |             |
     v             v             v
Qwen3 4B       Template or   Template and
Q4_K_M         targeted      referral
on GPU         question
     |
     v
Grounded answer with citations
```

The architecture avoids multi-LLM routing. A normal supported question requires no more than **one generator call**.

---

## 4. Model Selection

### 4.1 Answer generator

#### Primary candidate: Qwen3 4B

Recommended deployment configuration:

- GGUF `Q4_K_M` as the initial deployment format;
- `Q5_K_M` as a quality-versus-memory comparison;
- thinking mode disabled for ordinary helpdesk requests;
- temperature between 0.0 and 0.2;
- effective context limit of approximately 2,048–4,096 tokens;
- maximum response length of approximately 256–400 tokens;
- one active GPU generation at a time.

Qwen3 4B is the primary candidate because of its instruction-following and answer quality at a relatively small model size.

#### Baseline candidate: Llama 3.2 3B Instruct

Llama 3.2 3B should be retained as an experimental baseline because it may provide:

- lower latency;
- lower VRAM consumption;
- a useful comparison for groundedness and citation behavior.

The two generators should be tested separately. They should not be loaded into VRAM simultaneously during production deployment.

### 4.2 Embedding model

#### Primary candidate: `BAAI/bge-small-en-v1.5`

Recommended when the corpus and user questions are predominantly English:

- approximately 33 million parameters;
- 384-dimensional embeddings;
- practical CPU inference;
- relatively low RAM use;
- suitable for semantic retrieval over university documents.

Document embeddings are generated during ingestion. At request time, only the short user query is embedded.

#### Latency-focused alternative: `sentence-transformers/all-MiniLM-L6-v2`

This model should be evaluated when CPU latency is the highest priority:

- approximately 22 million parameters;
- 384-dimensional embeddings;
- fast CPU execution;
- broad library support.

#### Multilingual alternative

If the evaluation corpus contains substantial multilingual content, evaluate `intfloat/multilingual-e5-small`. A larger embedding model should only be introduced if the smaller model fails the retrieval-quality requirements.

Large embedding models such as a 0.6B model are not recommended as the initial deployment choice because their additional CPU latency and memory may not provide a proportional retrieval benefit.

### 4.3 Reranking model

#### Candidate: `cross-encoder/ms-marco-MiniLM-L-6-v2`

The reranker runs on CPU and receives only a small candidate set, such as the top 5–10 passages after rank fusion. It returns the best three passages for generation.

Reranking is conditional rather than mandatory. It may be skipped when retrieval confidence is already high. Three configurations must be compared:

1. no reranker;
2. reranker always enabled;
3. reranker enabled only for uncertain queries.

The production configuration will be selected from measured retrieval quality and latency rather than assumed superiority.

### 4.4 Intent and evidence-routing models

Routing must not use another generative language model. Recommended options are:

- deterministic rules for privacy, access, greetings, and obvious out-of-scope requests;
- fixed retrieval thresholds for the first implementation;
- calibrated logistic regression for learned evidence-sufficiency routing.

These components have negligible memory requirements compared with the generator and run entirely on CPU.

---

## 5. Knowledge Base Design

Only approved institutional documents are indexed. Each passage should preserve metadata including:

- document title;
- source or issuing office;
- document type;
- section and page;
- version;
- effective and expiration dates;
- authority level;
- access classification;
- extraction-quality status;
- canonical source identifier.

The ingestion process performs:

1. document validation;
2. text and table extraction;
3. cleaning without removing meaningful structure;
4. section-aware chunking;
5. metadata assignment;
6. duplicate and obsolete-document handling;
7. BM25 indexing;
8. embedding generation and vector indexing.

Poor extraction, outdated documents, and conflicting versions must be detected because routing quality cannot compensate for an unreliable corpus.

---

## 6. Retrieval Design

### 6.1 Hybrid retrieval

Each eligible question is searched using two complementary methods:

- **BM25** for exact terms, policy names, course codes, office names, fees, dates, and acronyms;
- **dense retrieval** for paraphrases and semantic similarity.

Both systems run on CPU. Metadata filters exclude inaccessible, invalid, or obsolete sources before evidence is sent to the generator.

### 6.2 Rank fusion

Reciprocal Rank Fusion combines the BM25 and dense rankings without requiring score normalization. This operation is deterministic and computationally inexpensive.

### 6.3 Candidate limits

An initial configuration is:

- retrieve up to 10 candidates from BM25;
- retrieve up to 10 candidates from dense search;
- fuse and retain up to 10 candidates;
- optionally rerank those candidates;
- send the strongest 3 passages to the generator.

Limits must be tuned using validation data. Sending more passages is not automatically better because it increases prompt latency and may distract a small generator.

---

## 7. Routing Design

### 7.1 Stage 1: Policy and access routing

Deterministic rules reject or redirect requests involving unauthorized personal, confidential, or student-specific information. Access metadata is enforced before retrieval results reach the generator.

Possible outcomes are:

- continue to retrieval;
- request authentication or authorization where supported;
- refuse access and provide an appropriate institutional referral.

### 7.2 Stage 2: Intent and query routing

Lightweight rules identify:

- greetings and basic conversational messages;
- clearly out-of-domain requests;
- university-information questions;
- incomplete or context-dependent follow-up questions.

A weak retrieval result must not automatically classify a question as casual conversation. Retrieval weakness is handled by evidence routing.

### 7.3 Stage 3: Evidence-sufficiency routing

After retrieval and optional reranking, the router selects one of four outcomes:

1. **Answer** — evidence is sufficient, valid, and non-conflicting.
2. **Clarify** — the question is ambiguous and a targeted detail could resolve it.
3. **Abstain** — the corpus does not support a reliable answer.
4. **Escalate** — authoritative sources conflict or the request needs human review.

Clarification and abstention responses should normally be templated, avoiding a generator call.

---

## 8. Small Calibrated Evidence Model

A small calibrated evidence model is **not a language model**. It is a lightweight statistical classifier that estimates whether the retrieved passages contain enough evidence to answer the question safely.

### 8.1 Candidate inputs

The classifier can use numerical and Boolean features such as:

- highest BM25 score;
- highest dense-retrieval score;
- highest reranker score;
- difference between the first and second result;
- agreement between BM25 and dense rankings;
- number of independently supporting passages;
- source authority and validity;
- question-to-evidence entity overlap;
- numeric or date overlap;
- conflicting-source indicator;
- query ambiguity indicators.

### 8.2 Candidate output

A binary model may estimate:

$$P(\text{evidence is sufficient})$$

A multiclass model may instead estimate probabilities for `answer`, `clarify`, and `abstain`. Explicit conflict rules can override the statistical output and route to escalation.

Illustrative thresholds are:

```text
Probability >= 0.75       -> Answer
Probability from 0.40     -> Clarify when ambiguity can be resolved
  to less than 0.75
Probability < 0.40        -> Abstain
Source conflict detected  -> Escalate or abstain
```

These values are placeholders. They must be selected from validation results.

### 8.3 Calibration

Calibration makes the classifier's probability meaningful. If a group of predictions receives approximately 80% confidence, approximately 80% of those cases should truly have sufficient evidence.

Calibration may use held-out validation data and a method such as Platt scaling or isotonic regression. Thresholds should then be chosen according to the cost of unsupported answers, unnecessary abstentions, and unnecessary clarifications.

### 8.4 Training data

Create labelled examples containing:

- question;
- retrieved candidate metadata and scores;
- expected route;
- evidence-sufficiency label;
- ambiguity label;
- conflict label.

The labels should be reviewed using a written annotation guide. The model must be trained only after enough representative routing examples are available.

### 8.5 Implementation progression

The recommended progression is:

1. begin with deterministic rules and manually tuned thresholds;
2. log retrieval features and reviewed route outcomes;
3. train logistic regression on the collected labels;
4. calibrate it on held-out data;
5. compare learned routing against the rule-based baseline;
6. deploy it only if it improves the accuracy-latency trade-off.

Logistic regression requires negligible CPU and RAM and is suitable for the constrained environment.

---

## 9. Prompt and Answer Policy

The generator receives:

- a concise system instruction;
- the user question;
- at most the strongest three evidence passages;
- source identifiers needed for citations;
- only the minimum necessary conversation context.

The generator must:

- answer only from supplied evidence;
- state uncertainty instead of inventing information;
- cite the supporting source and location;
- keep the answer concise and directly relevant;
- avoid exposing restricted content;
- avoid treating general model knowledge as institutional fact.

Generation settings should be conservative. Thinking mode and long chain-of-thought-style output should not be used for ordinary helpdesk responses.

After generation, deterministic checks should verify that:

- cited source identifiers exist in the retrieved set;
- required citations are present;
- the answer is within the output limit;
- no inaccessible source was used.

If citation validation fails, the system should retry once with a stricter format or abstain rather than return an unsupported answer.

---

## 10. Resource Allocation

| Component | Preferred hardware | VRAM impact |
|---|---|---:|
| Qwen3 4B or Llama 3.2 3B, quantized | GPU | Primary VRAM consumer |
| BM25 index | CPU/RAM | None |
| Dense vector index | CPU/RAM | None |
| BGE-small or MiniLM embedder | CPU/RAM | None |
| Reciprocal Rank Fusion | CPU | None |
| MiniLM reranker | CPU/RAM | None |
| Rules and logistic regression | CPU | None |
| Documents and metadata | Disk/RAM | None |

Only one generator should be loaded at a time. CPU models may be quantized or exported to an optimized runtime such as ONNX if benchmarks demonstrate a latency benefit.

---

## 11. Latency Strategy

The end-to-end latency budget consists of:

1. policy and intent checks;
2. query embedding;
3. BM25 and vector search;
4. rank fusion;
5. optional reranking;
6. evidence routing;
7. prompt processing;
8. answer generation.

Latency is controlled by:

- precomputing document embeddings;
- keeping indexes local;
- limiting retrieval candidates;
- reranking conditionally;
- supplying only the strongest evidence;
- constraining context and output length;
- disabling extended thinking;
- keeping the quantized generator resident in VRAM;
- allowing one active generation and queuing additional requests;
- caching only safe, non-personal, version-aware responses where appropriate.

The reranker and generator are expected to be the most significant optional and mandatory latency contributors, respectively.

---

## 12. Concurrency Strategy

An 8 GB GPU should not be assumed to support many simultaneous generation requests without degraded latency or memory pressure.

The initial production policy is:

- one active generation request;
- a bounded first-in, first-out queue;
- request timeout and cancellation handling;
- retrieval may proceed on CPU while the GPU is busy if CPU capacity permits;
- queue depth and waiting time are exposed as operational metrics.

Continuous batching may be evaluated only if the selected inference runtime supports it reliably within the VRAM limit.

---

## 13. Evaluation Plan

### 13.1 Generator comparison

Compare Qwen3 4B and Llama 3.2 3B under equivalent prompts, evidence, context limits, and quantization levels.

Measure:

- answer correctness;
- groundedness;
- citation correctness and completeness;
- unsupported-claim rate;
- instruction following;
- answer conciseness;
- time to first token;
- tokens per second;
- end-to-end latency;
- peak VRAM and RAM.

### 13.2 Embedding comparison

Compare at least:

- `BAAI/bge-small-en-v1.5`;
- `sentence-transformers/all-MiniLM-L6-v2`.

If multilingual retrieval is required, include `intfloat/multilingual-e5-small`.

Measure:

- Recall@k;
- Mean Reciprocal Rank;
- nDCG@k;
- query-embedding latency;
- retrieval latency;
- RAM usage.

### 13.3 Retrieval and reranking comparison

Compare:

- BM25 only;
- dense only;
- hybrid retrieval;
- hybrid plus always-on reranking;
- hybrid plus conditional reranking.

The reranker is retained only if its quality improvement justifies its latency.

### 13.4 Routing comparison

Compare:

- manually tuned rules and thresholds;
- calibrated logistic regression.

Measure:

- answer-route precision;
- unsafe-answer rate;
- appropriate-abstention rate;
- unnecessary-abstention rate;
- clarification precision;
- escalation accuracy;
- routing latency.

A false `answer` decision should receive a higher penalty than an unnecessary clarification because unsupported institutional guidance presents greater risk.

### 13.5 Quantization comparison

Compare at least `Q4_K_M` and `Q5_K_M` for the selected generator. If resources permit, include a higher-precision reference.

Measure quality, speed, context capacity, and peak VRAM on the actual 8 GB GPU.

---

## 14. Recommended Initial Configuration

The first feasible prototype should use:

```text
Generator:        Qwen3 4B GGUF Q4_K_M on GPU
Generator baseline: Llama 3.2 3B Instruct Q4_K_M
Embedding:        BAAI/bge-small-en-v1.5 on CPU
Sparse retrieval: BM25 on CPU
Fusion:           Reciprocal Rank Fusion on CPU
Reranker:         Disabled initially
Routing:          Rules plus validation-derived thresholds
Evidence:         Top 3 valid passages
Context:          2,048–4,096 effective tokens
Output:           256–400 tokens maximum
Concurrency:      One active GPU generation plus bounded queue
```

After this baseline is measured:

1. add conditional `ms-marco-MiniLM-L-6-v2` reranking;
2. compare retrieval quality and latency;
3. collect reviewed routing labels;
4. train and calibrate logistic regression;
5. deploy additions only when they provide measurable improvement.

---

## 15. Feasibility Decision

The architecture is feasible for an 8 GB VRAM environment because:

- only one quantized 3B–4B generator occupies significant VRAM;
- embedding and reranking models run on CPU;
- routing uses rules or logistic regression instead of another LLM;
- document embeddings are precomputed;
- prompt context and output length are bounded;
- unsupported questions are clarified or rejected without generation;
- concurrent GPU work is controlled by a queue.

Feasibility must ultimately be demonstrated through measurements on the target hardware. The final model, quantization, reranking policy, and routing thresholds should be selected from the measured balance of answer quality, retrieval accuracy, latency, RAM, VRAM, and concurrency behavior.