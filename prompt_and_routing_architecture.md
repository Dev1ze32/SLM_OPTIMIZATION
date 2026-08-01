# Prompt and Routing Architecture

## Purpose

This document defines the simple request-handling policy for the prototype. It supports the BM25-RAG, English-only, prototype-only study described in [THESIS_BATTLE_PLAN.md](THESIS_BATTLE_PLAN.md).

## Request flow

```text
English query
  -> scope check
  -> BM25 retrieval for in-scope query
  -> evidence check
  -> supported: one RAG generation call with citations
  -> unsupported/out of scope: predefined non-answer or office referral
```

## Routing rules

| Condition | Action | Generator call |
| --- | --- | --- |
| Clearly outside university-helpdesk scope | Return brief scope message | 0 |
| In-scope query with relevant LMS evidence | Generate a concise cited answer | 1 |
| In-scope query without sufficient LMS evidence | Return non-answer and office referral | 0 |

Do not classify a query as casual conversation merely because BM25 retrieves weak results. A university-related query can fail retrieval because the corpus is incomplete or its wording differs from the document.

## Prompt contract

Use one concise system instruction with the evidence supplied by the application:

```text
You are an offline university-helpdesk prototype.
Answer only using the supplied LMS evidence.
Do not invent policies, dates, fees, requirements, contacts, or procedures.
Give a concise English answer and cite the supplied document title and page or section.
If the evidence does not support an answer, do not guess.
```

QLoRA may reinforce this behavior, but the prompt remains necessary. It is not a promise that the model can never produce an unsupported statement.

## Evidence envelope

The application should send only retrieved passages and metadata needed for citations:

```json
{
  "document_id": "LMS_001",
  "title": "Student Handbook",
  "source": "University LMS / Student Handbook",
  "revision_date": "available when listed",
  "page_or_section": "Section 4.2",
  "text": "retrieved passage"
}
```

## Output contract

The prototype should produce an internal result in one of these forms:

```json
{
  "decision": "answer",
  "answer": "...",
  "citations": ["LMS_001"]
}
```

```json
{
  "decision": "refer",
  "answer": "I could not verify this from the selected LMS materials. Please contact the appropriate university office.",
  "citations": []
}
```

The application validates that every cited identifier belongs to evidence actually supplied to the model before displaying the response.

## Evaluation of routing and citations

- **Routing:** Was the supported query answered, and was the unsupported or out-of-scope query referred appropriately?
- **Citation validity:** Does the citation identify a supplied LMS passage?
- **Citation accuracy:** Does the cited passage support the nearby answer claim?
- **Groundedness:** Is the answer based on the retrieved evidence rather than unsupported model knowledge?

