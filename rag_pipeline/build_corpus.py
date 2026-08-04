"""
Stage 4: loads data/chunks/*.jsonl into routing/data/corpus.json, in the
schema routing/retriever.py's Passage dataclass expects (document_id,
title, source, revision_date, page_or_section, text).

The chunker's output has no document_id, title, source, or revision_date
-- those are routing-side citation concerns, not chunking concerns -- so
this stage is where they get assigned:

  document_id     "<PREFIX>-<0001>", prefix per source document, 4-digit
                   index within that document. Fixed-width and prefix-safe
                   (no prefix is a substring of another) so a citation
                   validator can check document membership by prefix
                   without accidentally crossing document boundaries.
  title, source   from DOCUMENT_META, keyed by the chunker's source_file.
  revision_date   "not available" -- none of the source PDFs carry a
                   machine-readable revision date. Flagged here rather
                   than guessed; ArchitectureAndDesign.md's rule is to
                   record the gap, not treat the answer as fully dated.
  page_or_section section_path when the chunker found one (343/364
                   chunks), else just the printed page number/range.
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
OUT_PATH = BASE_DIR.parent / "routing" / "data" / "corpus.json"

# source_file (as written by data_cleaning.py) -> (document_id prefix, title)
DOCUMENT_META = {
    "UC-PnC-Student-Handbook-Manual.pdf": ("HDBK", "UC-PnC Student Handbook Manual"),
    "UC-PnC-Internship-Manual.pdf": ("INTN", "UC-PnC Internship Manual"),
    "UC-PnC-Student-Research-and-Innovation-Manual.pdf": ("RSCH", "UC-PnC Student Research and Innovation Manual"),
}

_prefixes = [prefix for prefix, _ in DOCUMENT_META.values()]
assert len(_prefixes) == len(set(_prefixes)), "duplicate document_id prefix in DOCUMENT_META"
assert all(
    not a.startswith(b)
    for a in _prefixes
    for b in _prefixes
    if a != b
), "a document_id prefix is a substring of another -- citation checks would cross document boundaries"


def _page_or_section(chunk):
    if chunk["printed_page_start"] == chunk["printed_page_end"]:
        page = f"p. {chunk['printed_page_start']}"
    else:
        page = f"pp. {chunk['printed_page_start']}-{chunk['printed_page_end']}"
    section = chunk["section_path"]
    return f"{section} ({page})" if section else page


def _build_passage(chunk, prefix, title, index):
    return {
        "document_id": f"{prefix}-{index:04d}",
        "title": title,
        "source": f"University LMS / {title}",
        "revision_date": "not available",
        "page_or_section": _page_or_section(chunk),
        "text": chunk["text"],
    }


def build_corpus():
    passages = []
    summary = {}

    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        chunks = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not chunks:
            continue

        source_file = chunks[0]["source_file"]
        if source_file not in DOCUMENT_META:
            raise ValueError(
                f"No document_id prefix/title mapping for {source_file!r} "
                f"-- add it to DOCUMENT_META in {Path(__file__).name}"
            )
        prefix, title = DOCUMENT_META[source_file]

        for i, chunk in enumerate(chunks, start=1):
            passages.append(_build_passage(chunk, prefix, title, i))
        summary[path.name] = len(chunks)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"passages": passages}, f, ensure_ascii=False, indent=2)

    for name, count in summary.items():
        print(f"{name}: {count} chunks")
    print(f"TOTAL: {len(passages)} passages -> {OUT_PATH}")
    return summary


if __name__ == "__main__":
    build_corpus()
