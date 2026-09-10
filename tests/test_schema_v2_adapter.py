"""Tests for app/llm/schema_v2_adapter.py -- raw v2 JSON dict -> LLMOutput,
with a specific focus on `ambiguities` handling since that's the one piece
with no other home in the v2 schema (see the module docstring)."""
from __future__ import annotations

from app.llm.schema_v2_adapter import parse_v2_output


def test_structured_and_tools_parse_into_typed_lists():
    raw = {
        "intent": "FILTER_CANDIDATES", "replace_all": True,
        "structured": [{"field": "experience", "operator": "gte", "value": 5, "hard": True}],
        "tools": [{"raw_text": "Python", "match_mode": "exact", "hard": True}],
    }
    out = parse_v2_output(raw)
    assert out.intent == "FILTER_CANDIDATES"
    assert len(out.structured) == 1 and out.structured[0].field == "experience"
    assert len(out.tools) == 1 and out.tools[0].raw_text == "Python"
    assert out.filters == []  # v2 never populates the v1 field


def test_clarify_lookup_experience_search_fields_survive_untouched():
    raw = {"intent": "LOOKUP", "candidate_ref": "he", "lookup_field": "university"}
    out = parse_v2_output(raw)
    assert out.candidate_ref == "he"
    assert out.lookup_field == "university"

    raw2 = {"intent": "EXPERIENCE_SEARCH", "experience_query": "led a team"}
    assert parse_v2_output(raw2).experience_query == "led a team"


def test_ambiguities_become_the_clarify_question_when_none_given():
    raw = {"intent": "CLARIFY", "ambiguities": ["did you mean current or total experience?"]}
    out = parse_v2_output(raw)
    assert out.question == "did you mean current or total experience?"


def test_ambiguities_do_not_override_an_explicit_clarify_question():
    raw = {"intent": "CLARIFY", "question": "How many years?",
           "ambiguities": ["unrelated aside"]}
    out = parse_v2_output(raw)
    assert out.question == "How many years?"


def test_ambiguities_fold_into_message_for_non_clarify_intents():
    raw = {
        "intent": "FILTER_CANDIDATES", "replace_all": True,
        "structured": [{"field": "experience", "operator": "gte", "value": 5, "hard": True}],
        "ambiguities": ["not sure if 'senior' should also mean a title requirement"],
    }
    out = parse_v2_output(raw)
    assert "senior" in out.message
    assert len(out.structured) == 1  # real filters aren't lost alongside the note


def test_ambiguities_append_to_an_existing_message_rather_than_replacing_it():
    raw = {"intent": "UNSUPPORTED_FILTER", "message": "Salary data not available.",
           "ambiguities": ["also unclear what 'competitive package' means"]}
    out = parse_v2_output(raw)
    assert "Salary data not available." in out.message
    assert "competitive package" in out.message


def test_no_ambiguities_is_a_no_op():
    raw = {"intent": "FILTER_CANDIDATES", "replace_all": True}
    out = parse_v2_output(raw)
    assert out.message is None
    assert out.question is None
