"""Tests for the experience index -- ids, chunking, and above all the
separation contract: embedded text is the candidate's own words, classifier
output lives only in the classification store, and the two are joined by
`experience_id`.

Runs with no model server and no real data: the classifier and embedding
backends are never called here. The classifier's own behaviour is
`subdomain.py`'s concern; what these tests protect is the pipeline's
structure, which is the part a future change could silently break (e.g. by
"enriching" the embedded text with a predicted domain, which would make
semantic search circular -- see experience_text's module docstring).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import experience_index, experience_text
from app.core.experience_text import (
    chunk_id,
    chunk_text,
    classifier_sentences,
    embedding_text,
    experience_id,
    iter_experiences,
    parse_experience_id,
)


def _resume(n_experiences: int = 3) -> dict:
    return {
        "processId": "proc_test_1",
        "experience": [
            {
                "position": f"Backend Engineer {i}",
                "company": f"Acme {i}",
                "start_date": "January 2020",
                "end_date": "Present",
                "duration_months": 12,
                "is_ongoing": True,
                "description": (
                    f"• Built REST APIs in Python and PostgreSQL for service {i}. "
                    f"• Deployed containers to Kubernetes."
                ),
            }
            for i in range(n_experiences)
        ],
    }


# ---------------------------------------------------------------------------
# IDS
# ---------------------------------------------------------------------------

def test_ids_are_stable_and_round_trip():
    exp_id = experience_id("proc_abc", 2)
    assert exp_id == "proc_abc#exp2"
    assert parse_experience_id(exp_id) == ("proc_abc", 2)
    assert chunk_id(exp_id, 0) == "proc_abc#exp2#chunk0"


def test_parse_rejects_a_malformed_reference():
    # A bad reference must fail loudly rather than resolve to some other
    # candidate's record.
    with pytest.raises(ValueError):
        parse_experience_id("proc_abc")
    with pytest.raises(ValueError):
        parse_experience_id("proc_abc#chunk0")


def test_every_experience_gets_its_own_id():
    ids = [exp_id for exp_id, _, _ in iter_experiences(_resume(3))]
    assert ids == ["proc_test_1#exp0", "proc_test_1#exp1", "proc_test_1#exp2"]


def test_experience_with_no_text_is_skipped_not_indexed():
    resume = {"processId": "p", "experience": [{"position": "", "description": ""},
                                               {"position": "QA Analyst"}]}
    ids = [exp_id for exp_id, _, _ in iter_experiences(resume)]
    assert ids == ["p#exp1"]


def test_resume_with_no_process_id_yields_nothing():
    assert list(iter_experiences({"experience": [{"position": "Dev"}]})) == []


# ---------------------------------------------------------------------------
# EMBEDDING TEXT: original words only
# ---------------------------------------------------------------------------

def test_embedding_text_is_only_original_resume_fields():
    raw = _resume(1)["experience"][0]
    text = embedding_text(raw)
    assert "Backend Engineer 0" in text
    assert "Acme 0" in text
    assert "Built REST APIs in Python and PostgreSQL" in text


def test_embedding_text_excludes_any_classifier_or_derived_metadata():
    """The leak test. Whatever else changes, none of these may ever appear in
    the text handed to the embedding model."""
    raw = dict(_resume(1)["experience"][0])
    # Simulate a caller that stapled classifier output onto the raw dict --
    # embedding_text must read only the fields it knows are the candidate's.
    raw.update({
        "domain": "Engineering",
        "subdomain": "Backend Engineering",
        "predicted_subdomain": "Backend Engineering",
        "classification": {"domain": "Engineering"},
        "indicator_terms": ["rest apis"],
        "company_tier": "Tier1",
        "company_type": "Product",
    })
    text = embedding_text(raw)
    for leaked in ("Engineering", "Backend Engineering", "Tier1", "Product", "indicator"):
        assert leaked not in text, f"classifier/derived metadata leaked into embedded text: {leaked}"


def test_embedding_text_excludes_dates_and_durations():
    text = embedding_text(_resume(1)["experience"][0])
    assert "January 2020" not in text
    assert "12" not in text


def test_embedding_text_is_empty_for_an_empty_experience():
    assert embedding_text({}) == ""
    assert embedding_text({"position": "  ", "description": None}) == ""


# ---------------------------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------------------------

def test_one_chunk_per_experience_by_default():
    # The stated granularity: 3 experiences -> 3 chunks.
    chunks = [c for _, _, raw in iter_experiences(_resume(3))
              for c in chunk_text(embedding_text(raw))]
    assert len(chunks) == 3


def test_oversized_experience_splits_on_sentence_boundaries():
    sentence = "Built and shipped a distributed ingestion service in Go. "
    text = sentence * 200  # ~11k chars
    chunks = chunk_text(text, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    # No chunk starts or ends mid-word.
    assert all(c == c.strip() for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_chunking_never_drops_short_fragments():
    # The classifier's splitter discards fragments below MIN_SENTENCE_CHARS
    # as noise; chunking must not, or the candidate's own words go missing
    # from the embedded text.
    text = ("Led the team. ok. " + "Designed a distributed billing service in Go. " * 40)
    chunks = chunk_text(text, max_chars=500)
    assert "ok." in " ".join(chunks)
    assert "Led the team." in " ".join(chunks)


def test_a_single_sentence_longer_than_the_budget_is_still_split():
    chunks = chunk_text("x" * 2500, max_chars=1000)
    assert len(chunks) == 3
    assert sum(len(c) for c in chunks) == 2500


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


# ---------------------------------------------------------------------------
# SENTENCE SPLITTING (what the classifier reads)
# ---------------------------------------------------------------------------

def test_bullets_split_even_without_sentence_punctuation():
    text = "• Built REST APIs in Python • Deployed to Kubernetes clusters"
    assert classifier_sentences(text) == [
        "Built REST APIs in Python",
        "Deployed to Kubernetes clusters",
    ]


def test_short_fragments_are_dropped_as_noise():
    assert "ok." not in classifier_sentences("ok. Designed a data warehouse in Snowflake.")


def test_a_short_experience_still_yields_one_sentence():
    # Below MIN_SENTENCE_CHARS as a fragment, but it is the whole text --
    # dropping it would mean silently never classifying that experience.
    assert classifier_sentences("QA Analyst.") == ["QA Analyst."]


def test_sentence_count_is_capped():
    text = " ".join(f"Built service number {i} in Python." for i in range(100))
    assert len(classifier_sentences(text)) == experience_text.MAX_CLASSIFIER_SENTENCES


# ---------------------------------------------------------------------------
# THE TWO STORES
# ---------------------------------------------------------------------------

def _sample_stores():
    class_rows = [{
        "candidate_id": "proc_test_1",
        "experience_id": "proc_test_1#exp0",
        "experience_index": 0,
        "company": "Acme 0",
        "position": "Backend Engineer 0",
        "classification": {"domain": "Engineering", "subdomain": "Backend Engineering"},
    }]
    chunk_rows = [{
        "chunk_id": "proc_test_1#exp0#chunk0",
        "experience_id": "proc_test_1#exp0",
        "candidate_id": "proc_test_1",
        "experience_index": 0,
        "chunk_index": 0,
        "n_chunks": 1,
        "n_chars": 20,
        "text": "Backend Engineer at Acme.",
    }]
    return class_rows, chunk_rows


def test_stores_are_separate_files_joined_by_experience_id(tmp_path):
    class_rows, chunk_rows = _sample_stores()
    vectors = np.ones((1, 4), dtype=np.float32)

    experience_index.write_classifications(class_rows, tmp_path)
    experience_index.write_embeddings(chunk_rows, vectors, tmp_path)
    experience_index.write_manifest({"embedding_dim": 4}, tmp_path)

    labels = experience_index.load_classifications(tmp_path)
    chunks = experience_index.load_chunks(tmp_path)
    loaded = experience_index.load_vectors(tmp_path)

    assert np.array_equal(loaded, vectors)
    # The join: a vector row resolves to its classification via experience_id.
    assert labels[chunks[0]["experience_id"]]["classification"]["subdomain"] == "Backend Engineering"
    # And the chunk store itself carries no classifier output.
    assert "classification" not in chunks[0]
    assert set(chunks[0]) == {
        "chunk_id", "experience_id", "candidate_id", "experience_index",
        "chunk_index", "n_chunks", "n_chars", "text",
    }


def test_chunk_vector_misalignment_is_rejected(tmp_path):
    _, chunk_rows = _sample_stores()
    with pytest.raises(ValueError):
        experience_index.write_embeddings(chunk_rows, np.ones((2, 4), dtype=np.float32), tmp_path)


def test_classifications_group_by_candidate_in_resume_order(tmp_path):
    rows = [
        {"candidate_id": "a", "experience_id": "a#exp2", "experience_index": 2},
        {"candidate_id": "a", "experience_id": "a#exp0", "experience_index": 0},
        {"candidate_id": "b", "experience_id": "b#exp0", "experience_index": 0},
    ]
    experience_index.write_classifications(rows, tmp_path)
    grouped = experience_index.classifications_by_candidate(tmp_path)
    assert [r["experience_index"] for r in grouped["a"]] == [0, 2]
    assert list(grouped) == ["a", "b"]


def test_missing_index_reads_as_empty_not_an_error(tmp_path):
    assert experience_index.load_classifications(tmp_path) == {}
    assert experience_index.load_chunks(tmp_path) == []
    assert experience_index.load_manifest(tmp_path) == {}
