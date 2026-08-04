"""Pretty-print a sanitize/*.jsonl file for human review.

Reads one JSON record per line and rewrites it as indented, multi-line
JSON to a .pretty.json file next to it -- readable in any editor without
horizontal scrolling. The .jsonl itself is untouched; it stays one
record per line for the downstream chunker to consume.

Usage:
    python rag_pipeline/view_jsonl.py data/sanitize/UC-PnC-Student-Handbook-Manual.jsonl
    python rag_pipeline/view_jsonl.py data/sanitize/*.jsonl
"""
import json
import sys
from pathlib import Path


def convert(src):
    src = Path(src)
    dst = src.with_suffix(".pretty.json")
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    dst.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{src.name} ({len(records)} records) -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        convert(arg)
