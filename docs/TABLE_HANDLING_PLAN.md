# Table Handling — Implementation Plan

**Status: implemented and verified (see *Execution results* at the end).**

Revised plan for correcting table extraction in `rag_pipeline/data_cleaning.py`.
Supersedes the original "table-type classifier" spec. Read the *What changed*
section first — three items from that spec were dropped after verification.

## What changed from the original spec

| Original item | Status | Why |
| --- | --- | --- |
| 1. Table-type classifier with semantic signals | **Shrunk** | The discrimination logic already exists (`_grid_covers_region`, line 718). It just never gets a chance to run on 2-column tables. Root cause below. |
| 2. Flag ambiguous tables to a report | **Keep** | No flagging exists today. This is why the broken table went unnoticed for so long. |
| 3. Fix `�` encoding bug | **DROP — no bug** | Verified by codepoint: the characters are `•` (•) and `…` (…), both correct. The `�` was a terminal rendering artifact in the reviewer's console, not data. |
| 4. Re-run pipeline and verify | **Keep** | Now verifiable against the three reference images. |
| 5. Wire `tables` into `Passage` / evidence envelope | **DROP** | Every table's content is already in the `text` field (1018/1018 cells). Adding a structured field would duplicate every table in the prompt — a real cost for a 3B–4B model. Fixing the rendering removes the need. |
| **NEW** — recover indent hierarchy | **Add** | The reference images show 3 indent levels in the outlines that the extraction flattens. This is the one genuine data loss. |

## Root cause (verified, not inferred)

`TABLE_MIN_ROW_CELLS = 3` (line 303), enforced at line 595:

```python
if len(cells) >= TABLE_MIN_ROW_CELLS:
```

A 2-column table's rows have exactly 2 cells, so `detect_table_rows` can never
start a grid on one. Consequently, for any 2-column region:

1. `detect_table_rows` returns nothing → `probe` is empty (line 1012)
2. `_grid_covers_region` computes `covered / bands == 0`, below
   `GRID_PREFERENCE_COVERAGE = 0.30` → returns `False` (line 743)
3. The outline claim is therefore never filtered out (lines 1014–1015)
4. A genuine key→value table is emitted as `paired_outline` and rendered
   column-major by `render_paired_outline` (line 849)

`_grid_covers_region`'s own docstring describes exactly this key→value vs
parallel-outline problem and is designed to solve it. It is starved of input,
not wrong.

**Affected:** the internship `PROGRAM` → `# OF HOURS` table (11 programs, 11
hour values), rendered as two parallel lists so that answering "how many hours
for Computer Engineering?" requires positional counting.

## Changes

### 1. Give `_grid_covers_region` a fair probe on 2-column regions

Add an optional `min_row_cells` parameter to `detect_table_rows` (default stays
`TABLE_MIN_ROW_CELLS`, so global behavior is unchanged). In `clean_document`
(~line 1011), when testing an outline candidate, run a second probe restricted
to that candidate's y-range with `min_row_cells=2`, and pass that probe to
`_grid_covers_region`.

**Why this should discriminate correctly:** `detect_table_rows` fixes
`col_ranges` from the first row and requires later cells to fit within
`TABLE_COL_X_TOL = 15`. A key→value table has both columns at consistent x. The
paired outlines do *not* — their items sit at varying x precisely because they
are indented. The indentation being recovered in change 3 is the same signal
that separates the two shapes.

**This is a hypothesis grounded in reading the code, not a tested result.**
Verify it empirically before building on it: run the probe against the
internship table and against all 5 research outlines, and confirm it resolves
the former into 11 aligned rows and the latter into few or none.

If it does not discriminate cleanly, **stop and report** rather than loosening
`TABLE_MIN_ROW_CELLS` globally — that would create false tables throughout the
prose.

### 2. Flag ambiguous regions to a report

No table-level flagging exists today. Add it.

Collect a record for each 2-column region where the classification is not
clear-cut, and write it alongside the sanitize output (e.g.
`data/sanitize/table_report.json`) plus a one-line-per-entry stdout summary in
`process_all`.

Flag at minimum:

- a region whose grid coverage lands near `GRID_PREFERENCE_COVERAGE` (say 0.15–0.50)
- a region claimed as an outline whose two columns have *equal* item counts
  (the shape most likely to actually be a key→value table)
- any item produced by `_join_wrapped`, so wrap-merges can be spot-checked

Each record should carry: source file, pdf page, side, y-range, chosen type,
grid coverage, item counts, and the table `context` string.

The point is reviewability, not automation — a handful of flagged tables checked
by hand costs minutes. Prefer over-flagging to silence.

### 3. Recover indent hierarchy in `detect_paired_outline`

The geometry is available and currently discarded. `extract_lines` (line 85)
returns `(x0, y0, x1, y1, text)`, and `detect_paired_outline` already reads
`c[0]` at lines 811–812 — but only for the binary left/right split.

- Extend `_append_outline_item` (line 708) to carry each item's `x0` (and
  y-range, needed by change 1's verification) alongside its text.
- Within each column, cluster the items' `x0` values into indent levels using a
  tolerance in the spirit of `TABLE_COL_X_TOL`. Level = rank of the cluster.
- Store the depth per item in the `tables` metadata.
- Update `render_paired_outline` (line 849) to emit one item per line with
  leading indentation reflecting depth, instead of the current
  `"; ".join(...)` single-line flattening.

Multi-line output in the `text` field is already established precedent —
`render_table` (line 612) joins rows with `\n` for exactly this reason
("so column boundaries survive into the corpus"). Update
`render_paired_outline`'s docstring, which currently states the text field stays
single-line.

**Do not** attempt to fix the wrap-splitting under-merge. The docstring at lines
778–784 documents it as a deliberate choice — under-merging keeps every item and
never misfiles content, while a wrong merge silently corrupts. Indent data
constrains it (a *deeper* line is a child, never a wrap) but cannot resolve
same-indent wraps, since a wrapped continuation and a sibling sit at identical x.
Leave the existing conservative behavior and let change 2 flag the merges.

### 4. Re-run and verify against the reference images

```bash
wsl.exe -e bash -lc "cd /mnt/c/Users/HP-PAVILION/Downloads/ai_context/rag_pipeline && \
  ../venv/bin/python data_cleaning.py && \
  ../venv/bin/python chunk_documents.py && \
  ../venv/bin/python build_corpus.py"
```

The dependencies live in a **WSL venv** (`venv/`, Python 3.13). The Windows-side
Python 3.8 does not have them; `venv/bin/python` is a Linux binary and must be
invoked through WSL.

Confirm all three archetypes against the images in the repo root:

| Image | Archetype | Expected after fix |
| --- | --- | --- |
| `preview (2).webp` | 2-col key→value (PROGRAM / # OF HOURS) | Row-aligned: `Bachelor of Science in Computer Engineering \| 240 hours`. Currently the broken one. |
| `preview (1).webp` | 3-col grid (verb tenses by chapter) | Unchanged — already correct, including the blank Chapter IV proposal cell. Regression check. |
| `preview.webp` | Side-by-side outlines (IT Capstone) | Still column-major (correct — these do not pair row-wise), but with indent levels preserved. |

Also confirm the total stays at **364 chunks** (22 / 234 / 108) unless a change
is understood and intended, and re-check that `build_corpus.py` still produces
364 passages that construct `Passage` without field errors.

## Boundaries

- Files in scope: `rag_pipeline/data_cleaning.py` (primary),
  `rag_pipeline/chunk_documents.py` (verify multi-line rendering still splits
  correctly — it imports `render_paired_outline` and `render_table`).
- `build_corpus.py` should need no change; it reads `text`, not `tables`.
- Do **not** touch chunking granularity, embedding, retrieval, or anything under
  `routing/`.
- Do **not** invent row correspondences. Column-major plus a flag beats a wrong
  guess. This is the governing principle for every ambiguous case.

## Verified facts worth not re-deriving

- All 1018 table cells are present in the `text` field. Earlier reports of
  missing/truncated cells were an artifact of exact-substring matching across
  whitespace and quote normalization differences.
- Grid `tables` metadata is raw while `text` is normalized (`render_table` calls
  `normalize_text` only on the text side). Cosmetic; matters only if metadata
  ever reaches a prompt.
- The 5 research outlines are confirmed *not* row-aligned — the FINAL PAPER
  column inserts Certificate of Originality, Executive Summary, Acknowledgment,
  and Dedication, then the lists resync. Confirmed visually in `preview.webp`.
- 30 tables total across 27 chunks: 24 `grid`, 6 `paired_outline`.

## Execution results

All four changes implemented in `rag_pipeline/data_cleaning.py`. Verified by
running the full pipeline (`data_cleaning.py` → `chunk_documents.py` →
`build_corpus.py`) through the WSL venv and checking output against all three
reference images.

**Change 1 (grid vs. outline disambiguation).** Confirmed with a before/after
diff (`git stash`): the internship PROGRAM/HOURS table went `paired_outline` →
`grid` after the fix, and every one of the research manual's 10 genuine outline
records (5 tables × 2 page-halves) stayed `paired_outline`, unchanged — 8 grid
tables in that file also unchanged. No regression.

**Change 2 (flag report).** Implemented as `table_flags`, returned from
`clean_document` and written to `data/sanitize/table_report.json` by
`process_all`. First version flagged 65 regions — almost all false positives,
because "unmarked depth increase" fires on every ordinary chapter-heading →
first-subitem transition, which is the normal shape of these outlines, not an
error. Narrowed to only flag a *lone* single-item deeper span before the
outline returns to its prior depth (see `_unmarked_depth_increases`), which
dropped it to 17 flags, hand-checked:
- 13/14 `possible_unmerged_wrap` are genuine split phrases ("D. Research Ethics
  Review Committee" / "Evaluation", "I. Curriculum Vitae of Student" /
  "Researchers", etc.) — real instances of the documented, intentional
  under-merge.
- 1/14 (`Chapter IV RESULTS AND DISCUSSION` → lone `(Chapter Introduction)`) is
  a genuine judgment call, not a bug — correctly surfaced rather than guessed.
- 3 `grid_coverage_near_threshold(_2col)` flags, ratios 0.18–0.37 against the
  0.30 cutoff — all independently confirmed to have resolved to the correct
  type on inspection.
- All 17 flags came from the research manual; the other two documents produced
  zero, consistent with their simpler grid-only tables.

**Change 3 (indent recovery).** First attempt used `TABLE_COL_X_TOL` (15pt) as
the clustering tolerance and produced depth 0 for every item — flattened, no
better than before. Measured actual indent geometry directly (instrumented
`detect_paired_outline`): real steps are only ~5.7–8.5pt apart, smaller than
that tolerance. Introduced a separate `OUTLINE_INDENT_X_TOL = 3.0` for this
purpose; re-verified against `preview.webp`'s IT Capstone table and the
rendered indentation now matches the image (`Data Gathering Procedure` with
`Data Gathering Tools` / `Data Analysis Plan` / `System Development` nested
under it).

**Change 4 (re-run and verify).** All three archetypes confirmed in the final
`routing/data/corpus.json`:
- Grid (verb-tense table): unchanged, still correct.
- 2-col key→value (PROGRAM/HOURS): now row-aligned —
  `Bachelor of Science in Computer Engineering | 240 hours`.
- Side-by-side outlines: still column-major (no fabricated pairing), now with
  recovered indentation, verified to survive chunking intact in the final
  corpus passage (`RSCH-0050`).

Chunk counts unchanged at 364 total (22/234/108). All 364 passages in
`corpus.json` construct the `Passage` dataclass with no field errors and no
duplicate `document_id`s.

`build_corpus.py` and the evidence envelope were not touched, per the dropped
item 5 — `text` already carries every table correctly.
