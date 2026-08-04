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


def dedupe_blocks(blocks):
    """Drop blocks that are byte-identical in both position and text.

    Splitting a straddling block re-extracts a rectangle, which can also
    pick up a neighbouring block already collected through the normal
    path. Keying on rounded geometry plus text removes only those exact
    re-emissions, never two genuinely repeated lines at different spots."""
    seen, kept = set(), []
    for b in blocks:
        key = (round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1), b[4])
        if key in seen:
            continue
        seen.add(key)
        kept.append(b)
    return kept


def extract_lines(page, clip=None):
    """Per-visual-line text with geometry: (x0,y0,x1,y1,text).

    Finer-grained than get_text("blocks") -- MuPDF's block grouping merges
    adjacent table cells into one multi-line block (e.g. a row label and
    its four percentage cells come back as a single block, one cell per
    "\\n"), which loses which cell belongs to which column the moment the
    block's lines get flattened. Lines don't have that problem: each table
    cell is its own line with its own bbox, which is what column detection
    in detect_table_rows() needs. Unlike block-mode straddling on a spread,
    a `clip` rect cuts cleanly at line granularity -- no merged blocks
    spanning the midpoint, so no post-hoc dedup is needed here."""
    out = []
    d = page.get_text("dict", clip=clip) if clip is not None else page.get_text("dict")
    for blk in d["blocks"]:
        if blk.get("type") != 0:
            continue
        for line in blk["lines"]:
            text = "".join(s["text"] for s in line["spans"])
            if text.strip():
                x0, y0, x1, y1 = line["bbox"]
                out.append((x0, y0, x1, y1, text))
    return out


def extract_raw_pages(doc):
    """Returns a list of dicts: {physical_pdf_page, side, blocks, lines}
    where `blocks` is a list of (x0,y0,x1,y1,text) in natural reading order,
    already split by spread column when applicable, and `lines` is the same
    text at per-visual-line granularity (see extract_lines)."""
    raw = []
    for pno in range(doc.page_count):
        page = doc[pno]
        w, h = page.rect.width, page.rect.height
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        blocks = [b for b in blocks if b[6] == 0]  # text blocks only, drop images

        if w / h >= SPREAD_WIDTH_RATIO:
            mid = w / 2
            # A block belongs to a side by its horizontal center, EXCEPT
            # blocks that straddle the midpoint. Those are cases where
            # PyMuPDF merged text from BOTH half-pages into a single block
            # (a full-width footer, but also body text that happens to sit
            # at the same vertical band on both halves). Such a block must
            # be split, not assigned whole: re-extract its area one half at
            # a time so left-page text never leaks into the right logical
            # page, and neither half is emitted twice.
            left, right = [], []
            for b in blocks:
                x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
                center = (x0 + x1) / 2
                spans_both = x0 < mid - 20 and x1 > mid + 20
                if spans_both:
                    for side, rect in ((left, fitz.Rect(x0, y0, mid, y1)),
                                       (right, fitz.Rect(mid, y0, x1, y1))):
                        side.extend(s for s in page.get_text("blocks", clip=rect)
                                    if s[6] == 0)
                elif center < mid:
                    left.append(b)
                else:
                    right.append(b)

            left = dedupe_blocks(left)
            right = dedupe_blocks(right)
            left.sort(key=lambda b: (round(b[1], 0), b[0]))
            right.sort(key=lambda b: (round(b[1], 0), b[0]))
            left_lines = extract_lines(page, clip=fitz.Rect(0, 0, mid, h))
            right_lines = extract_lines(page, clip=fitz.Rect(mid, 0, w, h))
            raw.append({"pdf_page": pno + 1, "side": "L", "blocks": left,
                        "lines": left_lines, "page_height": h})
            raw.append({"pdf_page": pno + 1, "side": "R", "blocks": right,
                        "lines": right_lines, "page_height": h})
        else:
            blocks.sort(key=lambda b: (round(b[1], 0), b[0]))
            raw.append({"pdf_page": pno + 1, "side": None, "blocks": blocks,
                        "lines": extract_lines(page), "page_height": h})
    return raw


# ---------------------------------------------------------------------------
# Stage 2: cleaning
# ---------------------------------------------------------------------------

def mask_digits(text):
    return re.sub(r"\d+", "#", text.strip())


def detect_boilerplate(raw_pages, min_fraction=0.5):
    """Cross-page frequency detection of repeated headers/footers.
    A block's digit-masked text is boilerplate if it recurs on at least
    `min_fraction` of logical pages.

    Keys with no letters are skipped. A bare page number masks down to '#',
    which recurs everywhere and would then match any numeric-only content
    block -- a table cell, or the denominator of a formula. Page numbers in
    the margin are already removed by is_margin_noise()."""
    counts = Counter()
    n_logical_pages = len(raw_pages)
    for entry in raw_pages:
        seen_this_page = set()
        for b in entry["blocks"]:
            key = mask_digits(b[4])
            if not any(c.isalpha() for c in key):
                continue
            if key and key not in seen_this_page:
                counts[key] += 1
                seen_this_page.add(key)
    boilerplate = {k for k, c in counts.items() if c / n_logical_pages >= min_fraction}
    return boilerplate


def is_margin_noise(block, page_height):
    """Recognize page furniture missed by frequency detection.

    The manuals use a repeating branded footer, but the final notes spreads
    contain a footer whose left/right page numbers appear in one text block.
    That block occurs too rarely for cross-page frequency matching.  Restrict
    this fallback to the bottom margin and a narrow set of footer shapes so
    substantive text near the bottom of a page is never discarded.
    """
    _, y0, _, y1, text = block[:5]
    compact = re.sub(r"\s+", " ", text).strip()
    in_bottom_margin = y0 >= page_height * 0.84 or y1 >= page_height * 0.92
    if not in_bottom_margin:
        return False

    return bool(
        re.fullmatch(r"\d{1,4}(?:\s+University of Cabuyao)?(?:\s+\d{1,4})?", compact,
                     flags=re.IGNORECASE)
        or re.fullmatch(r"University of Cabuyao", compact, flags=re.IGNORECASE)
    )


# Capturing counterpart of is_margin_noise's first alternative: the digits
# discarded as boilerplate are the printed page number, the only page
# reference a reader holding the physical book can check a citation
# against ("pdf_page 40, side L" means nothing to them).
PRINTED_PAGE_PATTERN = re.compile(
    r"(\d{1,4})(?:\s+University of Cabuyao)?(?:\s+(\d{1,4}))?", re.IGNORECASE
)


def extract_printed_page(blocks, page_height, side):
    """Recover the printed page number from a footer block that
    is_margin_noise would otherwise discard outright.

    A straddling footer on a spread page is already re-extracted per half
    by extract_raw_pages, so most footer blocks here hold only one number
    -- this page's own. A few pages (the closing notes spreads) keep an
    un-split footer with both halves' numbers in one block ("38
    University of Cabuyao 39"); `side` picks the half that applies.
    """
    for b in blocks:
        if not is_margin_noise(b, page_height):
            continue
        compact = re.sub(r"\s+", " ", b[4]).strip()
        m = PRINTED_PAGE_PATTERN.fullmatch(compact)
        if not m:
            continue
        left, right = m.group(1), m.group(2)
        if right is not None:
            return int(right if side == "R" else left)
        return int(left)
    return None


SOFT_HYPHEN = "­"


def strip_soft_hyphen(text):
    """Drop a soft hyphen `normalize_text(keep_trailing_hyphen=True)` left
    behind once it is settled that nothing follows it to close up."""
    return text.replace(SOFT_HYPHEN, "")


def normalize_text(text, keep_trailing_hyphen=False):
    """Canonicalize one extracted fragment.

    `keep_trailing_hyphen` leaves a soft hyphen in place when it ends the
    fragment with nothing after it to close up. Callers that normalize a
    table cell BEFORE handing it to `_join_wrapped` need this: the half
    that completes the split word arrives as a separate fragment, so
    dropping the hyphen here destroys the only evidence that the two
    halves are one word -- "450 hours (industri­" + "al," came out as
    "(industri al,". Any fragment normalized this way must be run through
    `strip_soft_hyphen` once nothing more can follow it."""
    # Unicode normalization (curly quotes, NBSP, etc. -> canonical forms)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # A soft hyphen is a discretionary line-break hyphen. Where it survives
    # into extracted text it always marks a word split the layout engine
    # introduced ("Coun­\nseling"), never a real character.
    text = re.sub(r"­\s*(?=\S)" if keep_trailing_hyphen else r"­\s*", "", text)
    # Close up a line break after a hyphen but KEEP the hyphen. These
    # documents mark a word genuinely split by the layout engine with a
    # soft hyphen (handled just above), so a surviving ASCII hyphen at a
    # line end belongs to the text: "off-campus", "non-disclosure", and
    # form codes like "PNC:SDAS-FO-34". Dropping it silently welded those
    # into "offcampus" and "PNC:SDASFO-34".
    # \s* on both sides: the break can carry a trailing space, and when the
    # hyphen ends a block the block join adds a second newline, so the two
    # halves are not always separated by exactly one "\n". A form code split
    # that way ("(PNC:AA-" / "FO-28)") came out as "PNC:AA- FO-28)".
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", text)
    # Remaining newlines from block-internal line wraps become spaces
    text = text.replace("\n", " ")
    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


TABLE_ROW_Y_TOL = 2.5   # lines within this many pt of y0 are "the same row"
TABLE_COL_X_TOL = 15    # a continuation cell's x0 must be this close to a column to merge
TABLE_MAX_GAP = 20      # vertical gap (pt) beyond which a row can't continue the table
TABLE_MIN_ROWS = 2
TABLE_MIN_ROW_CELLS = 3


def _column_index(cell, col_ranges, tol=TABLE_COL_X_TOL):
    """Index of the column `cell` belongs to, or None if it fits none.

    Matched by horizontal overlap of the cell's x-range with the column's,
    not by left-edge proximity: a centered header ("RESEARCH" at 117-173)
    and the left-aligned data beneath it ("Adviser" at 93-123) start 24pt
    apart yet plainly share a column. Comparing left edges alone rejected
    the pairing and split every such table's header off as its own
    fragment. Cells that overlap nothing fall back to nearest-edge within
    `tol`, which covers a narrow cell sitting inside a wide column."""
    if not col_ranges:
        return None
    x0, x1 = cell[0], cell[2]
    best_i, best_overlap = None, 0.0
    for i, (a, b) in enumerate(col_ranges):
        overlap = min(b, x1) - max(a, x0)
        if overlap > best_overlap:
            best_i, best_overlap = i, overlap
    if best_i is not None:
        return best_i
    distance, i = min(
        (min(abs(x0 - a), abs(x0 - b), abs(x1 - a), abs(x1 - b)), i)
        for i, (a, b) in enumerate(col_ranges)
    )
    return i if distance <= tol else None


def _spans_multiple_columns(cell, col_ranges, min_overlap=3.0):
    """True when a line covers two or more of the table's columns.

    Body prose beneath a table runs the full text width, so it overlaps
    every column at once and _column_index() would still hand back one of
    them -- letting a paragraph be absorbed as a cell. A genuine cell sits
    within a single column, so straddling several marks the line as not
    belonging to the table at all."""
    hits = 0
    for a, b in col_ranges:
        if min(b, cell[2]) - max(a, cell[0]) >= min_overlap:
            hits += 1
    return hits >= 2


def _merge_into_row(row, cells):
    """Fold a continuation band's cells into the row above, matching each
    against the cell it sits under by horizontal overlap. A cell under
    nothing is appended, so a wrapped value in a column the previous row
    left empty still lands rather than being dropped."""
    for c in cells:
        best_i, best_overlap = None, 0.0
        for i, existing in enumerate(row):
            overlap = min(existing[2], c[2]) - max(existing[0], c[0])
            if overlap > best_overlap:
                best_i, best_overlap = i, overlap
        if best_i is None:
            row.append(c)
            continue
        x0, y0, x1, y1, text = row[best_i]
        row[best_i] = (min(x0, c[0]), y0, max(x1, c[2]), max(y1, c[3]),
                       _join_wrapped(text, c[4]))
    row.sort(key=lambda c: c[0])


def _slot_rows(rows):
    """Lay rows out on a shared set of columns, "" where a row has no
    value for one.

    Columns are taken from the row with the MOST cells, not the first
    row. A header often spans several data columns at once ("Academic
    Track" sitting over two percentage columns), so measuring columns
    from it collapses those data cells together -- a grading table's
    "25% | 25% | 35% | 20%" came out as "25% | 25% 35% | 20%". The widest
    row is the one that shows the table's real column count. Cells that
    match no column extend the layout rather than being discarded."""
    if not rows:
        return []
    reference = max(rows, key=len)
    col_ranges = [(c[0], c[2]) for c in reference]
    slotted = []
    for cells in rows:
        row = [""] * len(col_ranges)
        for c in cells:
            i = _column_index(c, col_ranges)
            if i is None:
                col_ranges.append((c[0], c[2]))
                row.append("")
                i = len(col_ranges) - 1
            row[i] = _join_wrapped(row[i], c[4]) if row[i] else c[4]
        slotted.append(row)
    width = len(col_ranges)
    return [r + [""] * (width - len(r)) for r in slotted]


def _is_continuation_band(cells, col_ranges, prev_bottom, prev_row):
    """Decide whether a band continues the previous row's wrapped text or
    starts a new row.

    Cell count alone can't tell these apart. A wrapped header wraps in
    EVERY column at once ("RESEARCH/PROPOSAL/FINAL" over
    "ENGAGEMENT/DEFENSE/DEFENSE"), so it has a full cell count yet is a
    continuation; conversely a row whose middle column is empty has a
    short cell count yet is a genuine new row ("Chapter IV" with no
    Proposal-column value). Judging by count merged the first case into
    two rows and the second case into the row above it.

    Two signals decide it instead:

    - A band that puts NOTHING in the key column (column 0) cannot be
      starting a new record, so it is always a continuation -- this is the
      common wrapped-value case ("present tense" under "Future tense or
      simple", "/ Capstone (for 2018 curriculum)" under "Project
      Feasibility Study").
    - A band that DOES occupy the key column is a new row unless its text
      vertically overlaps the row above (gap < 0). Real rows are laid out
      with clear separation, while the second line of a wrapped cell
      overlaps the first line's box. This is what separates a wrapped
      header from the data rows beneath it, and "Chapter IV" (a new row,
      clearly separated) from "if needed" (a wrapped key, overlapping).

    Every cell must also align to a known column; a band that introduces
    an unrecognized x position is treated as a new row rather than being
    force-fit into a column it doesn't belong to."""
    if not prev_row:
        return False
    if any(_column_index(c, col_ranges) is None for c in cells):
        return False
    if all(_column_index(c, col_ranges) != 0 for c in cells):
        return True
    # A bullet or enumerator in the key column opens a new list item. In
    # tables that give the marker its own narrow column, the previous
    # row's key cell is a bare "•", which the text test below would always
    # read as unfinished -- collapsing every pair of rows into one.
    key_cell = next((c for c in cells if _column_index(c, col_ranges) == 0), None)
    if key_cell is not None:
        key_text = normalize_text(key_cell[4])
        if BULLET_MARKER_PATTERN.match(key_text) or ENUMERATOR_PATTERN.match(key_text):
            return False
    if min(c[1] for c in cells) - prev_bottom < 0:
        return True
    # Fall back to the text tests: a key column left dangling mid-phrase
    # ("Statistician/Data Analyst,") is still an unfinished cell, and a
    # line that opens mid-phrase continues whatever came before it.
    prev_key = next((c for c in prev_row if _column_index(c, col_ranges) == 0), None)
    if prev_key is not None and _looks_incomplete(normalize_text(prev_key[4])):
        return True
    return all(_looks_like_continuation(normalize_text(c[4])) for c in cells)


def group_into_row_bands(lines, y_tol=TABLE_ROW_Y_TOL):
    """Cluster lines sharing a y-band into (y_repr, cells-sorted-by-x0)
    pairs, top to bottom. Shared by both table detectors below -- a row
    band is the basic unit either kind of table is built from."""
    lines_sorted = sorted(lines, key=lambda l: (l[1], l[0]))
    bands = []
    for ln in lines_sorted:
        if bands and abs(ln[1] - bands[-1][0]) <= y_tol:
            bands[-1][1].append(ln)
        else:
            bands.append([ln[1], [ln]])
    return [(y, sorted(cells, key=lambda l: l[0])) for y, cells in bands]


def nearest_preceding_context(blocks, y_start, max_gap=40):
    """The block immediately above a table, if close enough to plausibly
    be its caption/heading (e.g. "d. For Information Technology Capstone
    Project" sitting right above a Proposal/Final outline pair, or "g. The
    weight of the components shall be as follows:" above a grading table).
    Generic across any table shape -- it's just "what's the nearest text
    above this y-range", not tied to a specific heading wording.

    A heading that wraps onto a second line arrives as two blocks, so the
    walk continues upward through blocks that sit flush against the one
    below (no blank line between them) and joins them. Taking only the
    single nearest block truncated such headings to their last line --
    "b. For Industrial Engineering Researches (Project Feasibility" plus
    "Studiies" was reported as just "Studiies"."""
    candidates = sorted((b for b in blocks if b[3] <= y_start + 1),
                        key=lambda b: b[3], reverse=True)
    if not candidates:
        return None
    nearest = candidates[0]
    if y_start - nearest[3] > max_gap:
        return None

    # Walk up only far enough to pick up a heading's own wrapped second
    # line: consecutive lines of one heading sit flush (sub-half-line
    # gap), whereas a heading and the paragraph above it are a full line
    # apart. Capped at two blocks so a densely-set list above a table
    # (a table of contents, say) can't chain into the label.
    parts = [nearest]
    for b in candidates[1:2]:
        line_height = max(parts[-1][3] - parts[-1][1], 1)
        # abs(): consecutive lines of one heading can overlap by a hair as
        # well as touch, so the block above may end a fraction below where
        # this one starts.
        if abs(parts[-1][1] - b[3]) <= line_height * 0.5:
            parts.append(b)
    label = normalize_text("\n".join(p[4] for p in reversed(parts)))
    return label or None


def detect_table_rows(lines):
    """Find grid-shaped regions in a page's lines and return them as
    structured rows, so the caller can render cells with a delimiter
    instead of losing column boundaries to a flat space-joined string.

    A band of >=3 lines sharing a y-range opens a table: real prose is one
    line per y-band (a wrapped paragraph stacks vertically, it doesn't
    place 3+ independent text fragments side by side), so this generalizes
    across any document without hardcoding column positions. Once a table
    is open, whether each following band is a new row or the continuation
    of a wrapped cell is decided by _is_continuation_band() -- which is
    what lets a header wrapping across every column ("RESEARCH/PROPOSAL/
    FINAL" over "ENGAGEMENT/DEFENSE/DEFENSE") merge into one row while a
    sparse-but-real row ("Chapter IV", no Proposal value) stays separate.

    Column positions are taken from the first row, so a continuation is
    only merged into a column the table actually has. Rows are stored in
    column slots rather than as a bare list of whatever cells the band
    happened to contain: a row missing a value keeps an empty string in
    that position, so a cell's index always identifies its column.

    Returns a list of {"y_start", "y_end", "rows": [[cell_text, ...]]}.
    """
    bands = group_into_row_bands(lines)

    tables = []
    cur_rows = []       # each row: list of column slots, "" where empty
    col_ranges = []
    cur_y_start = None
    cur_y_end = None

    def flush():
        if len(cur_rows) >= TABLE_MIN_ROWS:
            tables.append({
                "y_start": cur_y_start,
                "y_end": cur_y_end,
                "rows": _slot_rows(cur_rows),
            })

    for y_repr, cells in bands:
        gap_ok = cur_rows and (y_repr - cur_y_end) <= TABLE_MAX_GAP
        aligned = gap_ok and all(
            _column_index(c, col_ranges) is not None
            and not _spans_multiple_columns(c, col_ranges)
            for c in cells)

        # Inside an open table, a band that lines up with the established
        # columns either extends the last row or starts a new one. A row
        # may be sparse -- "Chapter IV" with no Proposal-column value is
        # still a row -- so the full cell count isn't required; demanding
        # it previously ended tables early and dropped their later rows.
        #
        # A lone cell is the exception: it only joins as a continuation of
        # a wrapped cell, never as a new row. Single-cell bands between or
        # after tables are headings and prose ("ii. Graduate School Level"
        # separating two fee tables, a paragraph following the last row),
        # and admitting them as rows merged neighbouring tables together
        # and pulled trailing prose into the final row.
        if aligned:
            if _is_continuation_band(cells, col_ranges, cur_y_end, cur_rows[-1]):
                _merge_into_row(cur_rows[-1], cells)
                cur_y_end = max(cur_y_end, max(c[3] for c in cells))
                continue
            if len(cells) >= 2:
                cur_rows.append(list(cells))
                cur_y_end = max(cur_y_end, max(c[3] for c in cells))
                continue

        # A grouping header sitting inside a table ("EDUCATION" above the
        # education degrees, "ENGINEERING" above the engineering ones)
        # spans the columns rather than filling one, so it fails the
        # alignment test -- but ending the table there strands whatever
        # follows, and a group holding a single row was then dropped
        # entirely for falling under TABLE_MIN_ROWS. Kept as a key-column
        # row instead. A caption that introduces a NEW table is excluded
        # by its leading enumerator ("ii. Graduate School Level"), which
        # is what keeps the two fee tables apart.
        if cur_rows and gap_ok and col_ranges:
            heading = normalize_text(" ".join(c[4] for c in cells))
            first_word = heading.split()[0] if heading.split() else ""
            if (heading and uppercase_heading(heading)
                    and not ENUMERATOR_PATTERN.match(first_word)):
                # Pinned to the key column's x-range so slotting lands it
                # in column 0 regardless of how wide the heading runs.
                a, b = col_ranges[0]
                cur_rows.append([(a, cells[0][1], b, cells[-1][3], heading)])
                cur_y_end = max(cur_y_end, max(c[3] for c in cells))
                continue

        if len(cells) >= TABLE_MIN_ROW_CELLS:
            if cur_rows:
                flush()
                cur_rows = []
            cur_y_start = y_repr
            col_ranges = [(c[0], c[2]) for c in cells]
            cur_rows.append(list(cells))
            cur_y_end = max(c[3] for c in cells)
            continue

        flush()
        cur_rows = []

    flush()
    return tables


def render_table(table):
    """Cell text normalized individually; rows kept on separate lines and
    cells pipe-delimited so column boundaries survive into the corpus --
    unlike prose, which is deliberately flattened to one line by
    normalize_text's newline-to-space pass."""
    return "\n".join(
        " | ".join(normalize_text(cell) for cell in row)
        for row in table["rows"]
    )


OUTLINE_HEADER_MAX_WORDS = 6
OUTLINE_MIN_ITEMS = 4          # combined item count required to confirm a region
OUTLINE_CONNECTIVES = {
    "and", "or", "of", "the", "in", "to", "for", "with", "a", "an",
    "is", "are", "as", "at", "by", "from",
}
BULLET_MARKER_PATTERN = re.compile(r"^[••*\-]+$")
# "a." / "b)" / "iv." -- a list marker, i.e. the start of a new item.
ENUMERATOR_PATTERN = re.compile(r"^(?:[a-z]|[ivxlcdm]+)[.)]$", re.IGNORECASE)


def _looks_incomplete(text):
    """True when `text` reads like it's cut off mid-item and the next
    line most likely continues it rather than starting a new item.

    Outline items are usually short labels with no terminal punctuation
    at all ("Title Page", "Synthesis"), so "lacks a period" can't be the
    signal -- most complete items lack one. Only a small set of concrete
    cut-off markers count: a dangling connective word, a trailing hyphen,
    slash or comma (mid-phrase line break), an unclosed parenthesis, or
    the line being just a bullet glyph with no text of its own yet."""
    stripped = text.strip()
    if not stripped:
        return False
    if BULLET_MARKER_PATTERN.match(stripped):
        return True
    if stripped.endswith(("-", "/", ",")):
        return True
    if stripped.count("(") > stripped.count(")"):
        return True
    words = stripped.split()
    return bool(words) and re.sub(r"\W", "", words[-1]).lower() in OUTLINE_CONNECTIVES


def _looks_like_continuation(text):
    """True when `text` reads like the tail of the previous line rather
    than a fresh item -- judged from its own start instead of the previous
    line's end. Catches wraps the predecessor gives no hint about, e.g.
    "Project Feasibility Study" followed by "/ Capstone (for 2018
    curriculum)": the first line looks perfectly complete, only the
    leading "/" reveals the split."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(("/", ")", ",")):
        return True
    first = stripped.split()[0]
    # A list enumerator opens a new item even though it is lowercase --
    # "b. Student Research ..." is the next lettered clause, not the tail
    # of the row above. Covers letter and roman-numeral markers alike.
    if ENUMERATOR_PATTERN.match(first):
        return False
    return first[:1].islower()


def _join_wrapped(head, tail):
    """Join a wrapped cell's lines, closing up a word the layout engine
    split across them.

    Which of the two hyphens ends the line decides what to keep, exactly
    as in `normalize_text`:

    * a soft hyphen ("(industri­" + "al,") is the layout engine's own
      break mark and is not part of the word -- drop it and close up, so
      the cell reads "(industrial,". Joining with a space instead gave
      "(industri al,".
    * an ASCII hyphen ("Non-" + "Disclosure") belongs to the text, since
      these documents use the soft hyphen for their real word splits --
      keep it and close up, giving "Non-Disclosure". Dropping it welded
      the halves into "NonDisclosure".

    Either rule needs a word character in front of the hyphen. A cell
    holding nothing but a dash (the "1st offense | - | Reprimand"
    tables use one as a separator column) is not a broken word, and
    closing it up would swallow the separator into its neighbour.

    Everything else joins with a single space."""
    head, tail = head.rstrip(), tail.strip()
    if re.search(r"\w­$", head):
        return head[:-1] + tail
    if re.search(r"\w-$", head):
        return head + tail
    return re.sub(r"\s{2,}", " ", (head + " " + tail).strip())


def _append_outline_item(items, text):
    if items and (_looks_incomplete(items[-1]) or _looks_like_continuation(text)):
        items[-1] = _join_wrapped(items[-1], text)
    else:
        items.append(text)


GRID_PREFERENCE_COVERAGE = 0.30


def _grid_covers_region(grid_tables, lines, y_start, y_end):
    """Whether grid detection explains enough of a region for its row
    structure to be the better reading of it.

    Two-column regions come in two kinds and only one has row-by-row
    correspondence. A key->value table (degree -> required paper type)
    genuinely pairs across its columns, so its rows must be kept. Two
    parallel outlines (a Proposal-Paper and a Final-Paper outline of the
    same document) do not: they run to different lengths, so same-height
    items stop lining up as soon as one side gains an entry, and pairing
    them by row would invent correspondences that aren't in the source.

    Content alone can't separate them -- on a page where a table
    continues, the two outline columns are at different depths and share
    no wording, looking exactly like a mapping. Grid detection settles it
    instead: a real mapping resolves into aligned rows across much of the
    region, while parallel outlines yield almost none."""
    bands = len(group_into_row_bands(
        [l for l in lines if y_start - 1 <= l[1] and l[3] <= y_end + 1]))
    if not bands:
        return False
    covered = sum(len(t["rows"]) for t in grid_tables
                  if t["y_start"] >= y_start - 1 and t["y_end"] <= y_end + 1)
    return covered / bands >= GRID_PREFERENCE_COVERAGE


def _is_outline_header_band(cells):
    """A genuine header row ("PROPOSAL PAPER" / "FINAL PAPER"), not a
    coincidentally short body row ("Preliminaries" / "Preliminaries",
    "Title Page" / "Title Page"). Word-count alone can't tell these apart
    -- most outline items are just as short as a header. Visual case can:
    headers in this layout are the all-caps line, items are Title Case.
    Reuses uppercase_heading(), the same signal classify_page() already
    uses to spot section-divider headings elsewhere in this file."""
    if len(cells) != 2:
        return False
    left, right = normalize_text(cells[0][4]), normalize_text(cells[1][4])
    return (
        bool(left) and bool(right)
        and len(left.split()) <= OUTLINE_HEADER_MAX_WORDS
        and len(right.split()) <= OUTLINE_HEADER_MAX_WORDS
        and uppercase_heading(left) and uppercase_heading(right)
    )


def detect_paired_outline(lines):
    """Find two-column outline pairs: a short 2-cell header row (e.g.
    "PROPOSAL PAPER" / "FINAL PAPER") followed by a run of content split
    into two independent vertical lists by which side of the header's
    midpoint each line falls on.

    This is deliberately NOT row-paired -- unlike detect_table_rows, which
    assumes row N of every column is the same logical entry, the two
    outlines here can differ in length and content (one may have extra
    preliminary sections the other doesn't), so pairing them positionally
    would fabricate correspondences that aren't in the source. Each
    column is instead kept as its own ordered list under its header's own
    text, which is also what lets this generalize to any similarly-shaped
    two-column outline without hardcoding "Proposal"/"Final" wording.

    A wrapped continuation line is merged into the previous item in the
    SAME column only when it shows a concrete cut-off marker (see
    _looks_incomplete); anything else starts a new item. This can under-
    merge a wrap with no such marker (kept as two items instead of one) --
    intentional: guessing wrong would silently misfile content under an
    item it doesn't belong to, and every item is still captured, just
    possibly split. No text is ever dropped.

    Returns a list of {"y_start", "y_end", "headers": [h1, h2],
    "columns": {h1: [...], h2: [...]}}.
    """
    bands = group_into_row_bands(lines)
    results = []
    i = 0
    while i < len(bands):
        y_repr, cells = bands[i]
        if not _is_outline_header_band(cells):
            i += 1
            continue
        h_left, h_right = cells

        col_split = (h_left[0] + h_right[0]) / 2
        header_left = normalize_text(h_left[4])
        header_right = normalize_text(h_right[4])
        left_items, right_items = [], []
        y_end = max(h_left[3], h_right[3])
        j = i + 1
        while j < len(bands):
            by, bcells = bands[j]
            if by - y_end > TABLE_MAX_GAP:
                break
            if _is_outline_header_band(bcells):
                break  # the next pair's own header row
            left_cells = [c for c in bcells if c[0] < col_split]
            right_cells = [c for c in bcells if c[0] >= col_split]
            # keep_trailing_hyphen: an item wrapped mid-word ends this band
            # with a soft hyphen and is closed up by _join_wrapped when the
            # next band arrives. Whatever is left over is stripped below.
            if left_cells:
                _append_outline_item(
                    left_items,
                    " ".join(normalize_text(c[4], keep_trailing_hyphen=True)
                             for c in left_cells))
            if right_cells:
                _append_outline_item(
                    right_items,
                    " ".join(normalize_text(c[4], keep_trailing_hyphen=True)
                             for c in right_cells))
            y_end = max(y_end, max(c[3] for c in bcells))
            j += 1

        left_items = [strip_soft_hyphen(t) for t in left_items]
        right_items = [strip_soft_hyphen(t) for t in right_items]

        if len(left_items) + len(right_items) >= OUTLINE_MIN_ITEMS:
            results.append({
                "y_start": y_repr,
                "y_end": y_end,
                "headers": [header_left, header_right],
                "columns": {header_left: left_items, header_right: right_items},
            })
            i = j
        else:
            # Too short to confirm, or a key->value table rather than two
            # parallel outlines -- either way leave the bands for grid
            # detection, which keeps rows intact.
            i += 1

    return results


def render_paired_outline(table):
    """Single-line, order-preserving flattening for the `text` field
    (which stays fully flattened for every page, table or not); the
    structured {header: [items]} form lives in the page record's
    `tables` field for anything that needs the real list boundaries."""
    parts = []
    for header, items in table["columns"].items():
        parts.append(f"{header}: " + "; ".join(items))
    return " || ".join(parts)


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
NOTES_PATTERN = re.compile(r"^NOTES(?:\s+NOTES)?$", re.IGNORECASE)
FORM_CODE_PATTERN = re.compile(r"\bPNC:[A-Z]{2,6}-FO-\d{1,3}\b")

# Every chapter in these manuals opens the same way: "CHAPTER <roman> <TITLE>
# PNC:<code> Section <n>. <label>". Anchored to the page start (^, no
# MULTILINE) so it never fires mid-page -- the only other place "CHAPTER"
# appears is inside a rendered Proposal/Final outline table ("CHAPTER |
# PROPOSAL PAPER | FINAL PAPER"), which is never the first character of a
# page's text and whose next token isn't a roman numeral anyway.
CHAPTER_HEADING_PATTERN = re.compile(
    r"^(CHAPTER\s+[IVXLCDM]+\.?\s+.+?)(?:\s+PNC:|\s+Section\s+\d+\.|\Z)",
    re.IGNORECASE,
)
CHAPTER_HEADING_MAX_LEN = 80
# "Section 1. General Policies" / "Section 2. Specific Policies" repeat
# under every chapter, so the label alone doesn't identify a chunk -- it
# only becomes useful paired with the chapter/part above it in the
# breadcrumb. Captures the label together with its "Section n." prefix.
SECTION_HEADING_PATTERN = re.compile(
    r"(Section\s+\d+\.\s*[A-Z][A-Za-z /&\-]{2,40}?)(?=\s+\d+\.|\s+PNC:|\Z)"
)


def uppercase_heading(text):
    """True when a short record is a visual heading, not a short policy."""
    letters = [c for c in text if c.isalpha()]
    if not letters or len(text.split()) > 16:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.8


def classify_page(lines, joined_lower, logical_page_index=None):
    """Classify records for cleaning while retaining useful structure.

    ``section_divider`` and ``forms_index`` are retained for the future
    document-tree builder, but they are deliberately distinct from answerable
    prose so they cannot become standalone RAG chunks.
    """
    non_empty = [l for l in lines if l.strip()]
    joined = " ".join(non_empty).strip()

    # Checked before the length guard: a notes half-page carries the single
    # word "NOTES", which is shorter than the blank threshold.
    if NOTES_PATTERN.fullmatch(joined):
        return "notes"

    if len(joined) < 8:
        return "blank"

    form_codes = len(FORM_CODE_PATTERN.findall(joined))
    # A policy can legitimately cite several forms.  A forms index instead
    # has a much higher concentration of form codes than prose.
    if (joined.upper() == "LIST OF FORMS"
            or (form_codes >= 5 and form_codes / max(len(joined.split()), 1) >= 0.035)):
        return "forms_index"

    if re.fullmatch(r"Photos captured by .+", joined, flags=re.IGNORECASE):
        return "illustration"

    toc_hits = sum(1 for l in non_empty if any(p.search(l) for p in TOC_LINE_PATTERNS))
    toc_ratio = toc_hits / max(len(non_empty), 1)
    has_toc_keyword = any(l.strip().upper() in TOC_KEYWORDS for l in non_empty[:3])
    if toc_ratio >= 0.25 or has_toc_keyword:
        return "toc"

    near_start = logical_page_index is None or logical_page_index <= FRONT_MATTER_MAX_LOGICAL_PAGE
    if near_start and any(kw in joined_lower for kw in COPYRIGHT_KEYWORDS):
        return "front_matter"

    if (near_start and "MANUAL" in joined.upper() and "EDITION" in joined.upper()
            and uppercase_heading(joined)):
        return "cover"

    if uppercase_heading(joined):
        return "section_divider"

    return "content"


def clean_document(src):
    doc = fitz.open(src)
    raw_pages = extract_raw_pages(doc)
    boilerplate = detect_boilerplate(raw_pages)

    cleaned = []
    stats = Counter()
    # Label carried across logical pages: these tables run longer than one
    # half-page and repeat their header at the top of each continuation,
    # where there is no heading above them to read a label from. Without
    # this, roughly half of every program's outline came out untagged and
    # so couldn't be attributed to a program.
    last_table_label = {}
    # Breadcrumb state, carried across logical pages so a chunk that never
    # repeats its own chapter title (most pages after the chapter's first)
    # still knows which part/chapter/section it belongs to. `pending_part`
    # accumulates consecutive section_divider pages (some parts are
    # introduced by two divider pages in a row, e.g. "PART 1. ADMISSION &
    # ENROLLMENT" followed by "GUIDELINES") and is committed to
    # `current_part` on the next non-divider page.
    pending_part = []
    current_part = None
    current_chapter = None
    current_section = None
    for logical_idx, entry in enumerate(raw_pages, start=1):
        printed_page = extract_printed_page(
            entry["blocks"], entry["page_height"], entry["side"])
        all_blocks = [
            b for b in entry["blocks"]
            if mask_digits(b[4]) not in boilerplate
            and not is_margin_noise(b, entry["page_height"])
        ]

        raw_lines = []
        for b in all_blocks:
            raw_lines.extend(b[4].split("\n"))
        joined_raw = " ".join(raw_lines)
        page_type = classify_page(raw_lines, joined_raw.lower(), logical_page_index=logical_idx)
        stats[page_type] += 1

        content_lines = [
            l for l in entry["lines"] if not is_margin_noise(l, entry["page_height"])
        ]

        # Paired outlines (two independent lists side by side, e.g. a
        # Proposal-vs-Final-Paper section outline) claim their lines
        # before the grid detector runs for real. Without this, a band
        # where both columns happen to place a bullet + label side by side
        # (4 cells on one y-band) reads as a grid row and interleaves two
        # unrelated columns' fragments into one garbled row.
        #
        # A two-column region that grid detection can largely explain is
        # a row-correspondent table rather than two independent outlines,
        # so it is handed back to the grid path to keep its rows -- see
        # _grid_covers_region.
        probe = detect_table_rows(content_lines)
        outlines = [
            o for o in detect_paired_outline(content_lines)
            if not _grid_covers_region(probe, content_lines, o["y_start"], o["y_end"])
        ]
        consumed = {ln for o in outlines
                    for ln in content_lines
                    if o["y_start"] - 1 <= ln[1] and ln[3] <= o["y_end"] + 1}
        grid_tables = detect_table_rows(
            [l for l in content_lines if l not in consumed])

        tables = (
            [{"type": "grid", **t} for t in grid_tables]
            + [{"type": "paired_outline", **o} for o in outlines]
        )
        tables.sort(key=lambda t: t["y_start"])
        for t in tables:
            t["context"] = nearest_preceding_context(all_blocks, t["y_start"])
            key = tuple(t.get("headers") or ())
            if not key:
                continue
            if t["context"] is None:
                if key in last_table_label:
                    t["context"] = last_table_label[key]
                    t["continued"] = True
            else:
                last_table_label[key] = t["context"]

        # A block is only skipped from the prose join when it sits ENTIRELY
        # inside a table's y-range. A block whose bbox merely straddles a
        # table boundary is kept in full: MuPDF's block grouping can glue
        # a heading directly onto a table's first row (one block spanning
        # both), and excluding by center point would silently drop the
        # heading along with it. The worst case with full containment is a
        # table row's text appearing in both the table render and the
        # prose block that straddles it -- duplication, never loss.
        events = []
        for b in all_blocks:
            if any(t["y_start"] - 1 <= b[1] and b[3] <= t["y_end"] + 1 for t in tables):
                continue
            events.append((b[1], "block", b[4]))
        for t in tables:
            rendered = render_table(t) if t["type"] == "grid" else render_paired_outline(t)
            events.append((t["y_start"], "table", rendered))
        events.sort(key=lambda e: e[0])
        # Recorded for the downstream chunker: whether a page's last visual
        # element is a table decides which continuation signal applies when
        # deciding whether to stitch it to the next page -- a table's last
        # cell is often a bare word or percentage with no terminal
        # punctuation ("Reprimand", "20%"), which would misfire the prose
        # incompleteness heuristic if not told to defer to the table's own
        # `continued` marker instead.
        ends_with_table = bool(events) and events[-1][1] == "table"

        segments = []
        prose_buf = []
        for _, kind, content in events:
            if kind == "block":
                prose_buf.append(content)
            else:
                if prose_buf:
                    segments.append(normalize_text("\n".join(prose_buf)))
                    prose_buf = []
                segments.append(content)
        if prose_buf:
            segments.append(normalize_text("\n".join(prose_buf)))

        text = "\n".join(s for s in segments if s)

        if page_type == "section_divider":
            # A divider page announces its own title, not the part that
            # came before it -- without this branch a divider mid-document
            # showed the PRIOR part's breadcrumb (whatever `current_part`
            # was still set to), reading as if it belonged to the old part.
            pending_part.append(text)
            section_path = " > ".join(pending_part)
        else:
            if pending_part:
                current_part = " > ".join(pending_part)
                pending_part = []
                current_chapter = None
                current_section = None
            chapter_match = CHAPTER_HEADING_PATTERN.match(text)
            if chapter_match:
                new_chapter = chapter_match.group(1).strip()
                if len(new_chapter) > CHAPTER_HEADING_MAX_LEN:
                    new_chapter = new_chapter[:CHAPTER_HEADING_MAX_LEN].rsplit(" ", 1)[0]
                if new_chapter != current_chapter:
                    current_chapter = new_chapter
                    current_section = None
            section_matches = SECTION_HEADING_PATTERN.findall(text)
            if section_matches:
                current_section = section_matches[-1].strip()
            section_path = " > ".join(
                p for p in (current_part, current_chapter, current_section) if p
            )

        cleaned.append({
            "source_file": Path(src).name,
            "pdf_page": entry["pdf_page"],
            "side": entry["side"],
            "printed_page": printed_page,
            "page_type": page_type,
            "section_path": section_path,
            "char_count": len(text),
            "text": text,
            "ends_with_table": ends_with_table,
            "tables": [
                {k: v for k, v in t.items() if k not in ("y_start", "y_end")}
                for t in tables
            ],
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
