import json

import config
from retriever import Passage
from rag_pipeline_office import resolve_issuing_office, load_office_table


def test_resolve_prefers_exact_chapter_over_document_default():
    table = {
        "document_defaults": {"HDBK": "Student Affairs and Services Department"},
        "rules": [
            {"doc": "HDBK", "root": "PART 3. STUDENT SUCCESS",
             "chapter": "CHAPTER IX LIBRARY SERVICES", "office": "University Library"},
        ],
    }
    office = resolve_issuing_office(
        "HDBK-0150",
        "PART 3. STUDENT SUCCESS > CHAPTER IX LIBRARY SERVICES > Section 1. General Policies (pp. 61-62)",
        table,
    )
    assert office == "University Library"


def test_resolve_falls_back_to_document_default_for_front_matter():
    table = {"document_defaults": {"INTN": "Linkages Department"}, "rules": []}
    assert resolve_issuing_office("INTN-0001", "p. 4", table) == "Linkages Department"


def test_every_passage_has_a_known_office():
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        passages = json.load(f)["passages"]
    table = load_office_table()
    vocabulary = set(table["office_vocabulary"])

    assert len(passages) == 364
    unknown = [p["document_id"] for p in passages if p.get("issuing_office") in (None, "", "Unknown")]
    assert not unknown, f"{len(unknown)} passages have no office: {unknown[:10]}"
    off_vocab = {p["issuing_office"] for p in passages} - vocabulary
    assert not off_vocab, f"offices outside the closed vocabulary: {off_vocab}"


def test_every_passage_has_a_year():
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        passages = json.load(f)["passages"]
    missing = [p["document_id"] for p in passages if not p.get("year")]
    assert not missing, f"{len(missing)} passages have no year: {missing[:10]}"


def test_passage_loads_a_pre_migration_record():
    # issuing_office and year are defaulted, so an old corpus.json record
    # (built before this migration) must still load.
    p = Passage(
        document_id="HDBK-0001", title="t", source="s",
        revision_date="not available", page_or_section="p. 4", text="x",
    )
    assert p.issuing_office == "Unknown"
    assert p.year == "not available"
