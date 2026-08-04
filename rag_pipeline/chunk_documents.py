"""
Stage 3 (partial): turns sanitize/*.jsonl page records into retrieval-ready
chunks in two passes:

  1. Stitch pages that break a sentence, list, or table across a page
     boundary into one group (see should_stitch / _group_pages).
  2. Split each group back down at a size cap, but only at block
     boundaries a group already has -- a paragraph run, a list item, or a
     whole table -- never mid-sentence and never mid-table-row.

Running stitch-then-split instead of slicing fixed-size windows over the
raw per-page text is the point: a size-only chunker would just reproduce
the same mid-sentence tear at a different offset. Merging first means
splitting always starts from a complete unit and only cuts where the text
already had a seam.

Only page_type == "content" records are chunkable (see data_cleaning.py's
classify_page docstring); everything else is dropped here the same way it
already is at retrieval time.

Output: one <pdf_stem>.jsonl per source document into data/chunks/, each
line one chunk. Paths are self-relative like data_cleaning.py.
"""
import json
import re
from pathlib import Path

from data_cleaning import (
    BULLET_MARKER_PATTERN,
    CHAPTER_HEADING_PATTERN,
    OUTLINE_CONNECTIVES,
    render_paired_outline,
    render_table,
)

BASE_DIR = Path(__file__).resolve().parent
SANITIZE_DIR = BASE_DIR / "data" / "sanitize"
OUT_DIR = BASE_DIR / "data" / "chunks"

# Picked just above the pre-split median (1676 chars) so this trims the
# long tail (60/204 chunks over 3000 chars, max 8009) without reworking
# the bulk of chunks that were already a reasonable size.
CHUNK_SIZE_CAP = 1800

# ---------------------------------------------------------------------------
# Pass 1: stitching (unchanged from the page-merge stage)
# ---------------------------------------------------------------------------

# A page ending on one of these is a deliberate stop, not a tear: a full
# sentence, a quoted/parenthesized close, or (rare) a mid-list item that
# still reads as a complete clause.
PROSE_COMPLETE_ENDERS = (".", "!", "?", '"', "'", ")")
# Ending here always means more was coming: a dangling connector, an open
# parenthesis, a trailing list-item separator, or a colon introducing an
# enumeration that continues onto the next page.
PROSE_CONTINUATION_ENDERS = (",", "-", "/", ":", ";")


def _prose_tail_incomplete(text):
    """True when `text`'s own ending gives no sign it was meant to stop
    here -- the same signal set already proven for outline-item
    continuation in data_cleaning.py (_looks_incomplete), applied to a
    whole page's trailing prose instead of one cell.

    Deliberately has no "unrecognized ending, so assume complete"
    fallback: a well-formed page always ends on real punctuation, a
    bullet glyph, or (if genuinely done) nothing more to say. Stopping on
    a bare word ("...but are not limited") is exactly the failure mode
    being fixed, so an ending that matches none of the known patterns is
    treated as a cut, not given the benefit of the doubt.
    """
    tail = text.rstrip()
    if not tail:
        return False
    if tail.count("(") > tail.count(")"):
        return True
    last_line = tail.splitlines()[-1].strip()
    if BULLET_MARKER_PATTERN.match(last_line):
        return True
    last_char = tail[-1]
    if last_char in PROSE_CONTINUATION_ENDERS:
        return True
    if last_char in PROSE_COMPLETE_ENDERS:
        return False
    words = tail.split()
    last_word = re.sub(r"\W", "", words[-1]).lower() if words else ""
    if last_word in OUTLINE_CONNECTIVES:
        return True
    return True


# Three or more consecutive all-caps tokens opening a page: the manuals'
# universal way of starting a new titled unit ("MESSAGE FROM THE
# UNIVERSITY PRESIDENT", "STUDENT DEVELOPMENT & AUXILIARY SERVICES
# PHILOSOPHY & OBJECTIVES"). Needed as its own hard stop because
# _prose_tail_incomplete cannot see it from the PREVIOUS page: a signed
# message ends on its signature line ("HON. DENNIS FELIPE C. HAIN (Sgd)
# City Mayor/Chairman of the Board"), which carries no terminal
# punctuation and so reads as a mid-sentence cut, gluing four unrelated
# officials' messages into one chunk.
#
# Three tokens rather than two: single enumerators ("A.", "B.") and lone
# acronyms open plenty of genuine continuation pages, and a two-token
# threshold would swallow those.
NEW_HEADING_PATTERN = re.compile(r"^(?:[A-Z][A-Z0-9&/,.'\-]*\s+){2,}[A-Z][A-Z0-9&/,.'\-]*")


def _chapter_component(section_path):
    """Pull the "CHAPTER ..." segment out of a breadcrumb, or None if the
    page sits above chapter level (front matter, a part's own welcome
    page). Used as a hard stop: never stitch across a chapter change even
    if the trailing-prose heuristic would otherwise say to."""
    for part in (section_path or "").split(" > "):
        if part.upper().startswith("CHAPTER "):
            return part
    return None


def _table_key(table):
    return (table.get("type"), tuple(table.get("headers") or ()))


def _table_continues_into(cur, nxt):
    """True when the table finishing off `cur` is the SAME table picked
    back up at the top of `nxt` -- i.e. clean_document already tagged it
    `continued: True` because its header key matched one seen on an
    earlier page (see last_table_label in clean_document). This is the
    table-boundary signal: it identifies a specific continuing table
    rather than guessing from cell text, so a table whose last row just
    happens to end on a bare word never gets misread as cut off."""
    if not cur["ends_with_table"] or not cur["tables"] or not nxt["tables"]:
        return False
    first_next = nxt["tables"][0]
    return (
        first_next.get("continued") is True
        and _table_key(first_next) == _table_key(cur["tables"][-1])
    )


def _page_is_incomplete(record):
    if record["ends_with_table"]:
        # A table's last cell is routinely a bare word or a percentage
        # ("Reprimand", "20%") with no terminal punctuation -- judging it
        # by the prose heuristic would misfire on almost every table.
        # Whether the table itself continues is decided separately, by
        # _table_continues_into, once the next record is known.
        return False
    return _prose_tail_incomplete(record["text"])


def should_stitch(cur, nxt):
    if nxt is None:
        return False
    if nxt["source_file"] != cur["source_file"]:
        return False
    if CHAPTER_HEADING_PATTERN.match(nxt["text"]):
        return False  # next page opens a new chapter outright
    if NEW_HEADING_PATTERN.match(nxt["text"]):
        return False  # next page opens its own titled unit
    cur_chapter, nxt_chapter = _chapter_component(cur["section_path"]), _chapter_component(nxt["section_path"])
    if cur_chapter and nxt_chapter and cur_chapter != nxt_chapter:
        return False  # breadcrumb says a different chapter, even if headings didn't literally repeat
    return _page_is_incomplete(cur) or _table_continues_into(cur, nxt)


def _page_ref(record):
    return f"{record['pdf_page']}{record['side'] or ''}"


def _group_pages(records):
    """`records` is one document's sanitize output in original page order.
    Returns groups (lists of page records) of page_type == "content"
    records that belong in one chunk before size-cap splitting.

    Stitching only ever looks at the very next record in `records`, not
    the next content record -- a section_divider/toc/notes page sitting
    between two content pages is a hard stop regardless of how the prose
    heuristic reads, because it means a new topic was introduced there.
    Filtering to content-only records first (and losing that adjacency)
    let an unrelated later chapter get glued onto an earlier one whenever
    a run of divider pages sat between them and neither side happened to
    mention "CHAPTER" for the breadcrumb guard to catch.
    """
    groups = []
    i, n = 0, len(records)
    while i < n:
        if records[i]["page_type"] != "content":
            i += 1
            continue
        group = [records[i]]
        while (
            i + 1 < n
            and records[i + 1]["page_type"] == "content"
            and should_stitch(group[-1], records[i + 1])
        ):
            i += 1
            group.append(records[i])
        groups.append(group)
        i += 1
    return groups


# ---------------------------------------------------------------------------
# Pass 2: size-cap splitting, only at seams a group already has
# ---------------------------------------------------------------------------

# Two independent safe split points: after end punctuation, before
# whatever starts the next unit (a capital letter, a quote, or a fresh
# enumerated list number like "5.") -- and separately, before a bullet
# glyph wherever it falls, since a rendered "•" always marks a fresh list
# item even when the items themselves are ";"-separated with no sentence
# punctuation at all (a page-long run of "• ...; • ...; • ...;" would
# otherwise have no recognized split point and stay one oversized block).
#
# The list-number lookahead requires its OWN trailing period (\d+\.), not
# a bare digit: "R.A. 10627 Sec 2 letter d / R.A. 10175)" was torn apart
# right after "R.A." because a bare-digit lookahead reads any citation
# number after a period as a fresh list item starting -- "10627" is not
# one. Note a lookahead of "any capital letter after a period" still
# matches other abbreviations (initials, honorifics like "Dr."), which
# stay harmless as long as they get repacked into the same piece, but
# could in principle tear the same way if a cap boundary ever lands
# right on one; there's no abbreviation list here to rule those out.
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z"•]|\d+\.)|\s+(?=•)')


def _merge_table_pair(dst, src):
    """Combine continuation table `src` into `dst`, returning a new dict
    -- the same combining rule clean_document's own `continued` marking
    implies, applied here at block granularity instead of over a whole
    group's table list."""
    if dst["type"] == "grid":
        return {**dst, "rows": dst["rows"] + src["rows"]}
    if dst["type"] == "paired_outline":
        cols = {k: list(v) for k, v in dst.get("columns", {}).items()}
        for h, items in src.get("columns", {}).items():
            cols.setdefault(h, []).extend(items)
        return {**dst, "columns": cols}
    return dst


def _page_blocks(record, glue_to_previous=False):
    """Break one page record's flattened `text` back into the atomic
    units splitting is allowed to pack around: a grid table's rows
    collapse into ONE block (they can never be pulled apart by the size
    cap), a paired_outline table is already one line, and every other
    line is its own prose unit.

    Matched by re-rendering the record's OWN tables with the exact
    functions that produced `text` in clean_document, so a table's lines
    are identified exactly -- not guessed from pipe characters, which
    would misfire on any prose that happens to contain one.

    `glue_to_previous` marks the record's very first block as fused to
    whatever precedes it in the group -- set when this record only ended
    up in the group because should_stitch found its predecessor's tail
    incomplete or table-continuing. That specific seam is exactly the
    tear stitching exists to close, so the size-cap pass must never be
    allowed to cut there, even at the cost of a piece running over cap.
    """
    lines = record["text"].split("\n")
    tables = list(record["tables"])
    ref = _page_ref(record)
    blocks = []
    li, ti = 0, 0
    while li < len(lines):
        if ti < len(tables):
            t = tables[ti]
            rendered = render_table(t) if t["type"] == "grid" else render_paired_outline(t)
            rlines = rendered.split("\n")
            if lines[li:li + len(rlines)] == rlines:
                blocks.append({"text": rendered, "table": t, "page_refs": [ref]})
                li += len(rlines)
                ti += 1
                continue
        blocks.append({"text": lines[li], "table": None, "page_refs": [ref]})
        li += 1
    if blocks and glue_to_previous:
        blocks[0]["glue_before"] = True
    return blocks


def _combine_table_continuations(blocks):
    """Fold a continued table's two per-page renderings back into one
    atomic block before packing runs, so the size cap can never land
    between what is really one table's earlier and later rows."""
    combined = []
    for b in blocks:
        prev = combined[-1] if combined else None
        if (
            b["table"] is not None
            and prev is not None
            and prev["table"] is not None
            and b["table"].get("continued")
            and _table_key(b["table"]) == _table_key(prev["table"])
        ):
            merged_table = _merge_table_pair(prev["table"], b["table"])
            rendered = (
                render_table(merged_table) if merged_table["type"] == "grid"
                else render_paired_outline(merged_table)
            )
            combined[-1] = {
                "text": rendered,
                "table": merged_table,
                "page_refs": prev["page_refs"] + b["page_refs"],
                "glue_before": prev.get("glue_before", False),
            }
        else:
            combined.append(b)
    return combined


def _split_prose_block(block):
    """Break a prose block down to its own sentence-level units -- the
    finest safe grain available. Never applied to a table block: a table
    stays whole no matter what, rather than being broken mid-row.

    Deliberately does NOT look at the size cap and stops there: the one
    place that decides how much to regroup before flushing a piece is
    _pack_blocks, over ALL blocks in the group together (prose and
    table, across every page). Capping the size here too would have been
    redundant -- but worse, it silently skipped splitting any prose block
    that was under cap ON ITS OWN, even when several such blocks glued
    together (each individually fine, but chained by real continuations)
    added up past the cap with no split attempted anywhere in the run.
    """
    if block["table"] is not None:
        return [block]
    sentences = [s for s in SENTENCE_SPLIT_PATTERN.split(block["text"]) if s]
    if len(sentences) <= 1:
        return [block]
    out = [{"text": s, "table": None, "page_refs": list(block["page_refs"])} for s in sentences]
    if block.get("glue_before"):
        out[0]["glue_before"] = True  # only the first sentence inherits the seam to the previous page
    return out


def _pack_blocks(blocks, cap):
    """Greedily pack atomic blocks into size-capped pieces without ever
    splitting a block, and without ever starting a new piece right before
    a block marked `glue_before` -- that boundary is the exact seam
    should_stitch found unsafe, so it stays fused into the current piece
    even if that pushes the piece past the cap. A single block bigger
    than the cap on its own (an oversized table, or one sentence longer
    than the cap) becomes its own over-cap piece rather than being forced
    apart, the same exception."""
    pieces, cur, cur_len = [], [], 0
    for b in blocks:
        blen = len(b["text"])
        glued = b.get("glue_before", False)
        added = blen + (1 if cur else 0)  # +1 for the joining "\n"
        if cur and not glued and cur_len + added > cap:
            pieces.append(cur)
            cur, cur_len, added = [], 0, blen
        cur.append(b)
        cur_len += added
    if cur:
        pieces.append(cur)
    return pieces


def _finalize_piece(piece, group_by_ref, section_path, split_index, split_count):
    """Build one output chunk from a packed list of blocks.

    The breadcrumb is prepended to `text` itself (not just carried as a
    side field) on every piece, including the first: once a group is
    split, only piece 1 still contains the page's own opening heading
    words verbatim, so a later piece's own text would otherwise never
    mention "Chapter V ... Grading System" at all -- invisible to a BM25
    query on those terms despite genuinely belonging under that heading.

    Page-level identity (pdf_page/side/printed_page) stops meaning "this
    page" once a piece spans more than one, so it's a start/end pair plus
    `pages_spanned` instead -- a citation reads as "pp. 39-40", not a
    single page number covering only part of what's in it. char_count is
    the merged text's own length (breadcrumb included), not a sum, since
    joins and the prepended line both add characters of their own.
    """
    body = "\n".join(b["text"] for b in piece)
    text = f"{section_path}\n{body}" if section_path else body
    page_refs = []
    for b in piece:
        for pr in b["page_refs"]:
            if pr not in page_refs:
                page_refs.append(pr)
    first_rec, last_rec = group_by_ref[page_refs[0]], group_by_ref[page_refs[-1]]
    # Fall back past a page whose footer yielded no printed number, so one
    # unnumbered page in the middle of a span can't blank out a citation
    # the rest of the span could still support.
    printed = [group_by_ref[r]["printed_page"] for r in page_refs
               if group_by_ref[r]["printed_page"] is not None]
    tables = [b["table"] for b in piece if b["table"] is not None]
    return {
        "source_file": first_rec["source_file"],
        "pdf_page_start": first_rec["pdf_page"],
        "side_start": first_rec["side"],
        "pdf_page_end": last_rec["pdf_page"],
        "side_end": last_rec["side"],
        "printed_page_start": printed[0] if printed else None,
        "printed_page_end": printed[-1] if printed else None,
        "pages_spanned": len(page_refs),
        "stitched": len(page_refs) > 1,
        "page_refs": page_refs,
        "split_index": split_index,
        "split_count": split_count,
        "section_path": section_path,
        "char_count": len(text),
        "text": text,
        "tables": tables,
    }


def _chunk_group(group, cap):
    """Turn one stitched group of page records into one or more final
    chunks, splitting only at block boundaries -- never mid-sentence,
    never mid-table-row -- and only when the group is over `cap`.

    The cap used for packing is reserved down by the breadcrumb's own
    length, since every piece gets that breadcrumb prepended: without
    the reservation, adding it back on afterward would routinely push a
    piece already at the cap over it.
    """
    section_path = group[0]["section_path"] or group[-1]["section_path"]
    group_by_ref = {_page_ref(r): r for r in group}

    blocks = [
        b
        for i, r in enumerate(group)
        for b in _page_blocks(r, glue_to_previous=(i > 0))
    ]
    blocks = _combine_table_continuations(blocks)
    blocks = [sb for b in blocks for sb in _split_prose_block(b)]

    effective_cap = max(cap - len(section_path) - 1, 200) if section_path else cap
    pieces = _pack_blocks(blocks, effective_cap)
    return [
        _finalize_piece(piece, group_by_ref, section_path, idx + 1, len(pieces))
        for idx, piece in enumerate(pieces)
    ]


def chunk_document(records, cap=CHUNK_SIZE_CAP):
    """Public entry point: one document's sanitize records -> its chunks.
    Returns (chunks, per-document stats) so callers reporting progress
    don't have to re-derive the grouping and risk drifting from what this
    function actually did."""
    groups = _group_pages(records)
    per_group = [_chunk_group(g, cap) for g in groups]
    chunks = [c for pieces in per_group for c in pieces]
    stats = {
        "chunks": len(chunks),
        "groups": len(groups),
        "stitched_groups": sum(1 for g in groups if len(g) > 1),
        "split_groups": sum(1 for pieces in per_group if len(pieces) > 1),
        "over_cap": sum(1 for c in chunks if c["char_count"] > cap),
    }
    return chunks, stats


def process_all():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for path in sorted(SANITIZE_DIR.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        chunks, stats = chunk_document(records)

        out_path = OUT_DIR / path.name
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        print(f"{path.name}: {stats['chunks']} chunks from {stats['groups']} groups "
              f"({stats['stitched_groups']} multi-page, {stats['split_groups']} split by "
              f"size cap, {stats['over_cap']} still over {CHUNK_SIZE_CAP} chars) -> {out_path}")
        summary[path.name] = stats
    return summary


if __name__ == "__main__":
    process_all()
