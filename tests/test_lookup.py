"""Tests for LOOKUP: resolving a candidate reference and answering from
their real stored data -- never from the LLM's own words."""
from __future__ import annotations

from app.core.lookup import answer_lookup, resolve_candidate

CANDS = [
    {"id": "c1", "name": "Jay Sutaria", "university": ["New York University", "NMIMS University"],
     "college_tier": "High", "location": "Mumbai"},
    {"id": "c2", "name": "Rishi Shah", "university": ["KJ Somaiya School of Engineering"],
     "college_tier": "Low"},
]


def test_resolve_single_candidate_is_unambiguous():
    res = resolve_candidate(None, [CANDS[0]])
    assert res.candidate == CANDS[0]
    assert res.ambiguous_names is None


def test_resolve_by_name_among_several():
    res = resolve_candidate("Rishi", CANDS)
    assert res.candidate == CANDS[1]


def test_resolve_by_partial_name_skipping_middle_name():
    # "deep mehta" must match "Deep Paresh Mehta" -- a plain substring
    # check fails here since "Paresh" sits between the two words.
    cands = [
        {"id": "c1", "name": "Deep Paresh Mehta"},
        {"id": "c2", "name": "Rishi Shah"},
    ]
    res = resolve_candidate("deep mehta", cands)
    assert res.candidate == cands[0]


def test_resolve_no_ref_among_several_is_ambiguous():
    res = resolve_candidate(None, CANDS)
    assert res.candidate is None
    assert set(res.ambiguous_names) == {"Jay Sutaria", "Rishi Shah"}


def test_resolve_no_candidates_shown_yet():
    res = resolve_candidate("he", [])
    assert res.candidate is None
    assert res.ambiguous_names is None


def test_answer_uses_real_stored_value():
    answer = answer_lookup(CANDS[0], "university")
    assert "New York University" in answer
    assert "NMIMS University" in answer
    assert "Jay Sutaria" in answer


def test_answer_missing_data_is_honest():
    cand = {"id": "c3", "name": "Someone"}  # no university on file
    answer = answer_lookup(cand, "university")
    assert "don't have" in answer.lower()
    assert "Someone" in answer


def test_answer_unknown_field_is_honest():
    answer = answer_lookup(CANDS[0], "not_a_real_field")
    assert "don't track" in answer.lower()
