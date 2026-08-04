"""Tests for the stitch-then-split chunker.

Stdlib unittest, no external deps (the repo has no test runner configured):

    python rag_pipeline/test_chunk_documents.py

Two layers:
  * Unit tests over synthetic page records, so each rule is pinned
    independently of whatever the current PDFs happen to contain.
  * Invariant tests over the REAL sanitize/*.jsonl output, which is what
    actually catches regressions -- the synthetic cases only cover shapes
    someone thought to write down.
"""
import glob
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import chunk_documents as cd  # noqa: E402

SANITIZE_GLOB = str(BASE_DIR / "data" / "sanitize" / "*.jsonl")


def make_record(text, page=1, side="L", printed=None, section="PART 1 > CHAPTER I TEST",
                page_type="content", tables=None, ends_with_table=False):
    return {
        "source_file": "Test.pdf",
        "pdf_page": page,
        "side": side,
        "printed_page": printed if printed is not None else page,
        "page_type": page_type,
        "section_path": section,
        "char_count": len(text),
        "text": text,
        "ends_with_table": ends_with_table,
        "tables": tables or [],
    }


def body_of(chunk):
    """Chunk text minus the breadcrumb line the chunker prepends."""
    if chunk["section_path"]:
        return chunk["text"].split("\n", 1)[1]
    return chunk["text"]


def words(text):
    return Counter(re.findall(r"\S+", text))


class TestStitching(unittest.TestCase):
    def test_incomplete_tail_stitches_to_next_page(self):
        a = make_record("The requirements shall include, but are not limited")
        b = make_record("to the following items.", page=2)
        self.assertTrue(cd.should_stitch(a, b))

    def test_complete_sentence_does_not_stitch(self):
        a = make_record("The policy is hereby established.")
        b = make_record("A new topic begins here.", page=2)
        self.assertFalse(cd.should_stitch(a, b))

    def test_trailing_colon_stitches(self):
        a = make_record("The weight of the components shall be as follows:")
        b = make_record("Written work 25%.", page=2)
        self.assertTrue(cd.should_stitch(a, b))

    def test_unclosed_paren_stitches(self):
        a = make_record("Submit the form (see Appendix A for the full")
        b = make_record("list of requirements).", page=2)
        self.assertTrue(cd.should_stitch(a, b))

    def test_chapter_change_is_hard_stop(self):
        a = make_record("ends on a bare word", section="PART 1 > CHAPTER I ONE")
        b = make_record("more text", page=2, section="PART 1 > CHAPTER II TWO")
        self.assertFalse(cd.should_stitch(a, b))

    def test_explicit_chapter_heading_is_hard_stop(self):
        a = make_record("ends on a bare word")
        b = make_record("CHAPTER II STUDENT ENROLMENT PNC:AA-GU-19 Section 1. General Policies 1. Text.",
                        page=2)
        self.assertFalse(cd.should_stitch(a, b))

    def test_new_uppercase_heading_is_hard_stop(self):
        """A page opening with its own heading starts a new unit, even when
        the previous page ended on a signature line with no punctuation --
        otherwise distinct signed messages merge into one chunk."""
        a = make_record("HON. DENNIS FELIPE C. HAIN (Sgd) City Mayor/Chairman of the Board")
        b = make_record("MESSAGE FROM THE UNIVERSITY PRESIDENT It is with great pleasure.", page=2)
        self.assertFalse(cd.should_stitch(a, b))

    def test_non_content_page_breaks_the_group(self):
        recs = [
            make_record("ends on a bare word", page=1),
            make_record("DIVIDER", page=2, page_type="section_divider"),
            make_record("unrelated later text", page=3),
        ]
        groups = cd._group_pages(recs)
        self.assertEqual([len(g) for g in groups], [1, 1])

    def test_table_continuation_stitches(self):
        t1 = {"type": "grid", "headers": ["A", "B"], "rows": [["1", "2"]], "context": "T"}
        t2 = {"type": "grid", "headers": ["A", "B"], "rows": [["3", "4"]],
              "context": "T", "continued": True}
        a = make_record("A | B\n1 | 2", tables=[t1], ends_with_table=True)
        b = make_record("A | B\n3 | 4", page=2, tables=[t2], ends_with_table=True)
        self.assertTrue(cd.should_stitch(a, b))


class TestSplitting(unittest.TestCase):
    def test_group_under_cap_is_not_split(self):
        rec = make_record("One sentence. Two sentence.")
        pieces = cd._chunk_group([rec], cd.CHUNK_SIZE_CAP)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0]["split_count"], 1)

    def test_oversized_group_splits_at_sentence_boundaries(self):
        sentence = "This is a complete sentence of some length. "
        rec = make_record((sentence * 90).strip())
        pieces = cd._chunk_group([rec], cd.CHUNK_SIZE_CAP)
        self.assertGreater(len(pieces), 1)
        for p in pieces[:-1]:
            self.assertTrue(p["text"].rstrip().endswith("."),
                            f"piece ended mid-sentence: {p['text'][-60:]!r}")

    def test_citation_with_internal_periods_is_not_torn(self):
        """Regression: a bare-digit lookahead split "R.A. 10627" after
        "R.A." because it read the citation number as a new list item."""
        text = "Cyber-bullying done through electronic means. (R.A. 10627 Sec 2 letter d / R.A. 10175)"
        blocks = cd._split_prose_block({"text": text, "table": None, "page_refs": ["1L"]})
        joined = [b["text"] for b in blocks]
        self.assertFalse(any(t.rstrip().endswith("(R.A.") for t in joined),
                         f"citation was torn: {joined}")

    def test_enumerated_list_number_still_splits(self):
        text = "First item text here. 2. Second item text here. 3. Third item text here."
        blocks = cd._split_prose_block({"text": text, "table": None, "page_refs": ["1L"]})
        self.assertGreater(len(blocks), 1)
        self.assertTrue(any(b["text"].startswith("2.") for b in blocks))

    def test_table_is_never_split_even_over_cap(self):
        rows = [[f"row{i}", "x" * 60] for i in range(60)]
        t = {"type": "grid", "headers": ["A", "B"], "rows": rows, "context": "Big"}
        from data_cleaning import render_table
        rec = make_record(render_table(t), tables=[t], ends_with_table=True)
        pieces = cd._chunk_group([rec], cd.CHUNK_SIZE_CAP)
        self.assertEqual(len(pieces), 1, "an oversized table must stay in one piece")
        self.assertGreater(pieces[0]["char_count"], cd.CHUNK_SIZE_CAP)
        self.assertEqual(len(pieces[0]["tables"]), 1)
        self.assertEqual(len(pieces[0]["tables"][0]["rows"]), 60)

    def test_breadcrumb_prepended_to_every_piece(self):
        sentence = "This is a complete sentence of some length. "
        rec = make_record((sentence * 90).strip(), section="PART 2 > CHAPTER V GRADING")
        pieces = cd._chunk_group([rec], cd.CHUNK_SIZE_CAP)
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertTrue(p["text"].startswith("PART 2 > CHAPTER V GRADING\n"))

    def test_unsafe_page_seam_is_never_a_split_point(self):
        """The seam should_stitch judged unsafe must survive the size pass
        even when the combined group is over cap."""
        filler = "Filler sentence with words. " * 55
        a = make_record(filler + "the list shall include, but are not limited")
        b = make_record("to the items enumerated below.", page=2)
        group = [a, b]
        self.assertTrue(cd.should_stitch(a, b))
        pieces = cd._chunk_group(group, cd.CHUNK_SIZE_CAP)
        seam_piece = [p for p in pieces if "but are not limited" in p["text"]][0]
        self.assertIn("to the items enumerated below.", seam_piece["text"])

    def test_split_pieces_carry_their_own_subrange_metadata(self):
        long_tail = "Sentence number one here. " * 80
        a = make_record("opens the group and ends mid", page=1, side="L", printed=10)
        b = make_record(long_tail.strip(), page=2, side="R", printed=11)
        pieces = cd._chunk_group([a, b], cd.CHUNK_SIZE_CAP)
        self.assertGreater(len(pieces), 1)
        last = pieces[-1]
        self.assertEqual(last["page_refs"], ["2R"])
        self.assertEqual(last["printed_page_start"], 11)
        self.assertEqual(last["printed_page_end"], 11)
        self.assertEqual(last["pages_spanned"], 1)
        for idx, p in enumerate(pieces, start=1):
            self.assertEqual(p["split_index"], idx)
            self.assertEqual(p["split_count"], len(pieces))


class TestRealCorpusInvariants(unittest.TestCase):
    """Runs against whatever is currently in data/sanitize/."""

    @classmethod
    def setUpClass(cls):
        cls.docs = {}
        for path in sorted(glob.glob(SANITIZE_GLOB)):
            recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            cls.docs[Path(path).name] = recs
        if not cls.docs:
            raise unittest.SkipTest("no sanitize/*.jsonl to test against")

    def each_doc(self):
        for name, recs in self.docs.items():
            chunks, _stats = cd.chunk_document(recs)
            yield name, recs, chunks

    def test_no_words_lost(self):
        for name, recs, chunks in self.each_doc():
            with self.subTest(doc=name):
                original = Counter()
                for r in recs:
                    if r["page_type"] == "content":
                        original.update(words(r["text"]))
                produced = Counter()
                for c in chunks:
                    produced.update(words(body_of(c)))
                missing = original - produced
                # Table-continuation merging legitimately drops the duplicate
                # "PROPOSAL PAPER:" / "FINAL PAPER:" / "||" header framing
                # that appeared once per page before the two halves merged.
                header_framing = {"PROPOSAL", "FINAL", "PAPER:", "||"}
                real_loss = {w: n for w, n in missing.items()
                             if w not in header_framing and not w.rstrip(";") + ";" in produced}
                self.assertEqual(real_loss, {}, f"{name} lost words: {real_loss}")

    def test_no_piece_ends_mid_sentence(self):
        for name, recs, chunks in self.each_doc():
            with self.subTest(doc=name):
                for i, c in enumerate(chunks):
                    if c["split_count"] <= 1 or c["split_index"] == c["split_count"]:
                        continue
                    nxt = chunks[i + 1]
                    tail = c["text"].rstrip()
                    if c["tables"] or nxt["tables"]:
                        continue  # a table boundary is a legitimate seam
                    if body_of(nxt).lstrip().startswith("•"):
                        continue  # bullet boundary is a legitimate seam
                    self.assertRegex(
                        tail[-1], r"[.!?\"')]",
                        f"{name} {c['page_refs']} piece {c['split_index']}/"
                        f"{c['split_count']} ends mid-sentence: {tail[-70:]!r}")

    def test_every_record_table_survives_as_one_atomic_block(self):
        """If table re-matching ever fails, a grid table degrades into
        loose lines that the size cap CAN cut mid-row -- silently."""
        for name, recs, _ in self.each_doc():
            with self.subTest(doc=name):
                for r in recs:
                    if r["page_type"] != "content" or not r["tables"]:
                        continue
                    blocks = cd._page_blocks(r)
                    found = sum(1 for b in blocks if b["table"] is not None)
                    self.assertEqual(
                        found, len(r["tables"]),
                        f"{name} p{r['pdf_page']}{r['side']}: matched {found} of "
                        f"{len(r['tables'])} tables as atomic blocks")

    def test_table_rows_are_never_split_across_pieces(self):
        for name, recs, chunks in self.each_doc():
            with self.subTest(doc=name):
                for c in chunks:
                    for t in c["tables"]:
                        if t["type"] != "grid":
                            continue
                        rendered = cd.render_table(t)
                        self.assertIn(rendered, c["text"],
                                      f"{name} {c['page_refs']}: table not intact in its chunk")

    def test_breadcrumb_present_on_every_chunk(self):
        for name, recs, chunks in self.each_doc():
            with self.subTest(doc=name):
                for c in chunks:
                    if not c["section_path"]:
                        continue
                    self.assertTrue(
                        c["text"].startswith(c["section_path"] + "\n"),
                        f"{name} {c['page_refs']} missing breadcrumb")

    def test_page_metadata_consistent(self):
        for name, recs, chunks in self.each_doc():
            with self.subTest(doc=name):
                for c in chunks:
                    self.assertEqual(len(c["page_refs"]), c["pages_spanned"])
                    self.assertEqual(c["stitched"], c["pages_spanned"] > 1)
                    self.assertEqual(c["char_count"], len(c["text"]))
                    self.assertLessEqual(c["pdf_page_start"], c["pdf_page_end"])
                    if c["printed_page_start"] is not None and c["printed_page_end"] is not None:
                        self.assertLessEqual(c["printed_page_start"], c["printed_page_end"])

    def test_over_cap_chunks_are_justified(self):
        """Over-cap is allowed only for an atomic table or an unsafe seam
        that must not be cut -- never for ordinary packable prose."""
        for name, recs, chunks in self.each_doc():
            with self.subTest(doc=name):
                for c in chunks:
                    if c["char_count"] <= cd.CHUNK_SIZE_CAP:
                        continue
                    justified = bool(c["tables"]) or c["pages_spanned"] > 1
                    self.assertTrue(
                        justified,
                        f"{name} {c['page_refs']} is {c['char_count']} chars with no "
                        f"table and no cross-page seam to justify it")

    def test_chunks_never_span_two_chapters(self):
        for name, recs, chunks in self.each_doc():
            with self.subTest(doc=name):
                for c in chunks:
                    heads = re.findall(r"CHAPTER\s+[IVXLCDM]+\s+[A-Z]", body_of(c))
                    self.assertLessEqual(
                        len(heads), 1,
                        f"{name} {c['page_refs']} spans multiple chapters: {heads}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
