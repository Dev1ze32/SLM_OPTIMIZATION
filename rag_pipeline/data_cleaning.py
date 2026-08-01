"""
Stage 1 (Extraction) + Stage 2 (Cleaning) of KNOWLEDGE_BASE_INGESTION_PLAN.md.

Batch mode: reads every PDF in data/raw/ and writes one cleaned JSONL per
PDF into data/sanitize/, using paths relative to this script's own location
(so it works the same whether run from rag_pipeline/ or elsewhere):

    rag_pipeline/
      data/
        raw/        <- source PDFs go here
        sanitize/   <- cleaned_pages_<name>.jsonl written here
      data_cleaning.py   <- this file

Design notes (generalized, not hardcoded to one document):

- Source PDFs may be print-imposition exports: page 1 is a normal single
  page (front cover), but pages 2..N are each a two-page SPREAD (e.g.
  792x612pt = two 396x612 half-pages side by side). Naive extraction would
  interleave left-page and right-page text. We detect spreads by page
  width and split each one into a left and right logical page using
  PyMuPDF's block geometry (layout-aware, per the plan's Stage 1
  requirement). PDFs that aren't spreads (normal single pages) are handled
  the same way automatically -- no per-document config needed.
- Running headers/footers are detected generically by cross-page frequency
  of block text with digits masked out -- not hardcoded strings -- so the
  same logic works across differently-branded manuals.
- Front matter (copyright/credits pages) is only ever near the start of a
  document, so that classification is bounded to the first ~10 logical
  pages. This avoids misclassifying a real chapter that happens to share
  wording with a credits page (e.g. a document with an actual "Research
  Ethics Review Committee" chapter).
- No OCR fallback: pdffonts should be checked per-document to confirm a
  real text layer exists. If a source *is* scanned, ocr_fallback() below
  is a stub to wire in pytesseract per the plan.

Output: one <pdf_stem>.jsonl per source PDF, each line a logical page with
a `source_file` field identifying which PDF it came from. Ready to feed
into Stage 3 (structure-aware chunking into corpus.json), which is
intentionally NOT done in this script.
"""

import fitz  # PyMuPDF
import re
import json
import unicodedata
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "sanitize"
SPREAD_WIDTH_RATIO = 1.2  # width/height above this => treat page as a 2-up spread


# ---------------------------------------------------------------------------
# Stage 1: layout-aware extraction
# ---------------------------------------------------------------------------

def ocr_fallback(page):
    """Stub for scanned pages. Not used here -- pdffonts shows embedded,
    subsetted fonts on every page, i.e. a real text layer exists."""
    raise NotImplementedError(
        "This document has a text layer (see pdffonts); OCR fallback was "
        "intentionally not implemented for this run."
    )


def extract_raw_pages(doc):
    """Returns a list of dicts: {physical_pdf_page, side, blocks}
    where `blocks` is a list of (x0,y0,x1,y1,text) in natural reading order,
    already split by spread column when applicable."""
    raw = []
    for pno in range(doc.page_count):
        page = doc[pno]
        w, h = page.rect.width, page.rect.height
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        blocks = [b for b in blocks if b[6] == 0]  # text blocks only, drop images

        if w / h >= SPREAD_WIDTH_RATIO:
            mid = w / 2
            # A block belongs to a side by its horizontal center, EXCEPT
            # full-width footer/header blocks that straddle the midpoint --
            # those are handled separately below (kept whole, split by line).
            left, right, straddling = [], [], []
            for b in blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                center = (x0 + x1) / 2
                spans_both = x0 < mid - 20 and x1 > mid + 20
                if spans_both:
                    straddling.append(b)
                elif center < mid:
                    left.append(b)
                else:
                    right.append(b)

            left.sort(key=lambda b: (round(b[1], 0), b[0]))
            right.sort(key=lambda b: (round(b[1], 0), b[0]))
            raw.append({"pdf_page": pno + 1, "side": "L", "blocks": left,
                        "straddling": straddling})
            raw.append({"pdf_page": pno + 1, "side": "R", "blocks": right,
                        "straddling": straddling})
        else:
            blocks.sort(key=lambda b: (round(b[1], 0), b[0]))
            raw.append({"pdf_page": pno + 1, "side": None, "blocks": blocks,
                        "straddling": []})
    return raw


# ---------------------------------------------------------------------------
# Stage 2: cleaning
# ---------------------------------------------------------------------------

def mask_digits(text):
    return re.sub(r"\d+", "#", text.strip())


def detect_boilerplate(raw_pages, min_fraction=0.5):
    """Cross-page frequency detection of repeated headers/footers.
    A block's digit-masked text is boilerplate if it recurs on at least
    `min_fraction` of logical pages."""
    counts = Counter()
    n_logical_pages = len(raw_pages)
    for entry in raw_pages:
        seen_this_page = set()
        all_blocks = entry["blocks"] + entry["straddling"]
        for b in all_blocks:
            key = mask_digits(b[4])
            if key and key not in seen_this_page:
                counts[key] += 1
                seen_this_page.add(key)
    boilerplate = {k for k, c in counts.items() if c / n_logical_pages >= min_fraction}
    return boilerplate


def strip_straddling_footer(straddling_blocks, boilerplate):
    """Straddling blocks (e.g. 'L_NUM\\nUniversity of X\\nR_NUM') are
    footers/headers by construction on a spread; if their masked form is
    boilerplate, drop them entirely (they contribute no content)."""
    kept = []
    for b in straddling_blocks:
        if mask_digits(b[4]) in boilerplate:
            continue
        kept.append(b)
    return kept


def normalize_text(text):
    # Unicode normalization (curly quotes, NBSP, etc. -> canonical forms)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # De-hyphenate words split across a line break: "reprodu-\nction" -> "reproduction"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remaining newlines from block-internal line wraps become spaces
    text = text.replace("\n", " ")
    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


TOC_LINE_PATTERNS = [
    re.compile(r"^\d{1,4}\s*\t"),          # "12 \t Our History"
    re.compile(r"\.{2,}\s*\d{1,4}\s*$"),   # classic dot-leader TOC
    re.compile(r"^\d{1,4}\s{2,}\S"),       # "12   Our History"
]
TOC_KEYWORDS = {"TABLE OF CONTENTS", "CONTENTS", "PAGE"}
# Keywords specific enough to only ever appear on an actual copyright/
# credits page. Deliberately excludes generic terms like "publisher" or
# "review committee" -- those collide with substantive chapters (e.g. a
# manual with a real "Research Ethics Review Committee" chapter) and
# produced false positives when tested against a second document.
COPYRIGHT_KEYWORDS = [
    "copyright ©", "all rights reserved", "isbn", "layout artist",
]
# Front matter (copyright/credits pages) only ever occurs near the start
# of a document. Bounding the keyword check to early pages prevents a
# keyword match deep in the body from being misclassified.
FRONT_MATTER_MAX_LOGICAL_PAGE = 10


def classify_page(lines, joined_lower, logical_page_index=None):
    """Returns one of: 'toc', 'front_matter', 'blank', 'content'."""
    non_empty = [l for l in lines if l.strip()]
    if len(" ".join(non_empty).strip()) < 8:
        return "blank"

    toc_hits = sum(1 for l in non_empty if any(p.search(l) for p in TOC_LINE_PATTERNS))
    toc_ratio = toc_hits / max(len(non_empty), 1)
    has_toc_keyword = any(l.strip().upper() in TOC_KEYWORDS for l in non_empty[:3])
    if toc_ratio >= 0.25 or has_toc_keyword:
        return "toc"

    near_start = logical_page_index is None or logical_page_index <= FRONT_MATTER_MAX_LOGICAL_PAGE
    if near_start and any(kw in joined_lower for kw in COPYRIGHT_KEYWORDS):
        return "front_matter"

    return "content"


def clean_document(src):
    doc = fitz.open(src)
    raw_pages = extract_raw_pages(doc)
    boilerplate = detect_boilerplate(raw_pages)

    cleaned = []
    stats = Counter()
    for logical_idx, entry in enumerate(raw_pages, start=1):
        content_blocks = [b for b in entry["blocks"] if mask_digits(b[4]) not in boilerplate]
        straddling_kept = strip_straddling_footer(entry["straddling"], boilerplate)
        # straddling blocks are footers/headers on spreads; if not fully
        # boilerplate-matched, keep them appended (rare, but don't lose text)
        all_blocks = content_blocks + straddling_kept

        raw_lines = []
        for b in all_blocks:
            raw_lines.extend(b[4].split("\n"))
        joined_raw = " ".join(raw_lines)
        page_type = classify_page(raw_lines, joined_raw.lower(), logical_page_index=logical_idx)
        stats[page_type] += 1

        text = normalize_text("\n".join(b[4] for b in all_blocks))

        cleaned.append({
            "source_file": Path(src).name,
            "pdf_page": entry["pdf_page"],
            "side": entry["side"],
            "page_type": page_type,
            "char_count": len(text),
            "text": text,
        })

    return cleaned, stats, boilerplate


def process_all():
    """Process every PDF in data/raw/, writing one cleaned JSONL per
    document into data/sanitize/. Returns a summary dict for logging."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw PDF folder not found: {RAW_DIR}\n"
            f"Expected layout: <this script's folder>/data/raw/*.pdf"
        )

    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {RAW_DIR}")
        return {}

    summary = {}
    for pdf_path in pdf_paths:
        print(f"\n===== {pdf_path.name} =====")
        cleaned, stats, boilerplate = clean_document(src=str(pdf_path))

        print("Logical pages extracted:", len(cleaned))
        print("Page type breakdown:", dict(stats))
        print("Detected boilerplate (header/footer) patterns:")
        for b in sorted(boilerplate):
            print("  -", repr(b))

        out_path = OUT_DIR / f"{pdf_path.stem}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in cleaned:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        content_pages = [c for c in cleaned if c["page_type"] == "content"]
        content_chars = sum(c["char_count"] for c in content_pages)
        print(f"Wrote {len(cleaned)} logical pages -> {out_path}")
        print(f"Content pages retained for chunking: {len(content_pages)} "
              f"({content_chars:,} chars)")

        summary[pdf_path.name] = {
            "logical_pages": len(cleaned),
            "content_pages": len(content_pages),
            "content_chars": content_chars,
            "stats": dict(stats),
            "output": str(out_path),
        }

    return summary


if __name__ == "__main__":
    summary = process_all()

    if summary:
        print("\n===== Summary =====")
        total_content_pages = sum(s["content_pages"] for s in summary.values())
        total_chars = sum(s["content_chars"] for s in summary.values())
        for name, s in summary.items():
            print(f"  {name}: {s['content_pages']} content pages, {s['content_chars']:,} chars")
        print(f"  TOTAL: {total_content_pages} content pages, {total_chars:,} chars "
              f"across {len(summary)} documents")