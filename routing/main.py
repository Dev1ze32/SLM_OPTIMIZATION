"""
Demo entry point. Run interactively or against a fixed sample-query list
to see the routing decision and latency breakdown per gate.

Usage:
    python main.py            # interactive loop
    python main.py --demo     # run built-in sample queries (no typing needed)

Requires OPENAI_API_KEY to be set in the environment for the "answer"
path (Gate 3). The "out_of_scope" and "refer" paths work without it.
"""

import argparse

from router import Router


SAMPLE_QUERIES = [
    "How do I request a leave of absence?",            # expect: answer
    "What is the grading appeal process?",              # expect: answer
    "Can you recommend a good pizza place nearby?",     # expect: out_of_scope
    "What's the deadline to drop a course this term?",  # expect: answer
    "How do I get a refund for my parking permit?",     # expect: refer (likely no matching passage)
]


def print_result(query: str, result) -> None:
    print("=" * 72)
    print(f"QUERY:    {query}")
    print(f"DECISION: {result.decision}")
    print(f"ANSWER:   {result.answer}")
    if result.citations:
        print(f"CITATIONS: {result.citations}")
    print("-" * 72)
    print(f"scope_in_scope     : {result.scope_in_scope}")
    print(f"scope_similarity   : {result.scope_similarity}")
    print(f"bm25_scores        : {result.bm25_scores}")
    print(f"generator_calls    : {result.generator_calls}")
    print("-" * 72)
    print(f"scope latency (ms)      : {result.scope_latency_ms:.2f}")
    print(f"retrieval latency (ms)  : {result.retrieval_latency_ms:.2f}")
    print(f"generation latency (ms) : {result.generation_latency_ms:.2f}")
    print(f"TOTAL latency (ms)      : {result.total_latency_ms:.2f}")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo", action="store_true", help="Run built-in sample queries"
    )
    args = parser.parse_args()

    print("Loading router (embedding model + BM25 index)...")
    router = Router()
    print("Ready.\n")

    if args.demo:
        for q in SAMPLE_QUERIES:
            print_result(q, router.route(q))
        return

    print("Interactive mode. Type a query, or 'quit' to exit.\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        print_result(query, router.route(query))


if __name__ == "__main__":
    main()