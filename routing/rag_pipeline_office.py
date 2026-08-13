"""Objective 3's section -> issuing_office lookup, applied at corpus-build time.

The mapping is a researcher-constructed, versioned artifact: it makes the
manuals' existing organization-by-responsible-unit explicit. It is not
extracted from the source text (text-mining was tested and rejected: only
32% of passages mention anything office-like, and keyword matching pulls
false positives such as "Patent Office" from the IP chapter).
"""

import json
import os
import re

_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rag_pipeline", "data", "office_lookup.json",
)

_PAGE_SUFFIX = re.compile(r"\s*\((?:p\.|pp\.)[^)]*\)\s*$")


def load_office_table(path: str = _TABLE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_breadcrumb(page_or_section: str) -> list:
    """'A > B > C (pp. 4-5)' -> ['A', 'B', 'C']. Bare 'p. 4' -> ['p. 4']."""
    return [part.strip() for part in _PAGE_SUFFIX.sub("", page_or_section).split(">")]


def resolve_issuing_office(document_id: str, page_or_section: str, table: dict) -> str:
    doc = document_id[:4]
    parts = parse_breadcrumb(page_or_section)
    root = parts[0]
    chapter = parts[1] if len(parts) > 1 else parts[0]

    for rule in table.get("rules", []):
        if rule["doc"] == doc and rule["root"] == root and rule["chapter"] == chapter:
            return rule["office"]
    for rule in table.get("rules", []):
        if rule["doc"] == doc and rule["root"] == root:
            return rule["office"]
    return table.get("document_defaults", {}).get(doc, "Unknown")
