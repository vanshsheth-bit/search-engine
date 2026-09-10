"""Tests for app/core/taxonomy.py -- the v2 schema's resolution layer.

Requires the real merged_tools.json at repo root (same as
test_skill_taxonomy.py) -- skipped if it's not present in this checkout,
since resolve_filters reuses skill_taxonomy.py's real lookups rather than
mocking them (the whole point of this module is correctly reusing that
existing, tested logic, not reimplementing it).
"""
from __future__ import annotations

import os

import pytest

from app.core.taxonomy import active_filter_terms, resolve_filters
from app.models.schemas import Filter, LLMOutput, StructuredItem, ToolItem

_ROOT = os.path.join(os.path.dirname(__file__), "..")
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(_ROOT, "merged_tools.json")),
    reason="merged_tools.json not present in this checkout",
)


def test_structured_item_maps_directly_to_a_filter():
    out = LLMOutput(intent="FILTER_CANDIDATES", structured=[
        StructuredItem(field="experience", operator="gte", raw_text="5 years", value=5, hard=True),
    ])
    filters, skip_notes = resolve_filters(out, "5 years of experience")
    assert len(filters) == 1
    f = filters[0]
    assert (f.field, f.operator, f.value, f.hard) == ("experience", "gte", 5, True)
    assert skip_notes == []


def test_structured_item_preserves_hard_false():
    out = LLMOutput(intent="FILTER_CANDIDATES", structured=[
        StructuredItem(field="notice_period", operator="lte", raw_text="prefer immediate",
                       value=0, hard=False),
    ])
    filters, _ = resolve_filters(out, "prefer immediate joiners")
    assert filters[0].hard is False


def test_skill_experience_carries_the_skill_key():
    out = LLMOutput(intent="FILTER_CANDIDATES", structured=[
        StructuredItem(field="skill_experience", operator="gte", raw_text="5 years of Python",
                       value=5, skill="Python", hard=True),
    ])
    filters, _ = resolve_filters(out, "5 years of Python")
    assert filters[0].skill == "Python"


def test_exact_tool_only_canonicalizes_never_widens():
    # "React.js" is a known alias -> canonicalizes to "React", but must NOT
    # pull in related tools (Angular, Vue.js) the way "expand" mode would --
    # a demand for one specific tool must not match someone with a
    # different, merely-related one.
    out = LLMOutput(intent="FILTER_CANDIDATES", tools=[
        ToolItem(raw_text="React.js", match_mode="exact", hard=True),
    ])
    filters, _ = resolve_filters(out, "knows React.js")
    assert len(filters) == 1
    f = filters[0]
    assert f.field == "skill"
    assert f.operator == "contains"
    assert isinstance(f.value, str)


def test_expand_tool_widens_via_real_taxonomy():
    out = LLMOutput(intent="FILTER_CANDIDATES", tools=[
        ToolItem(raw_text="machine learning", match_mode="expand", hard=True),
    ])
    filters, _ = resolve_filters(out, "knows machine learning")
    assert len(filters) == 1
    f = filters[0]
    assert f.operator == "in"
    assert isinstance(f.value, list) and len(f.value) > 1
    assert "machine learning" in [v.lower() for v in f.value]


def test_expand_tool_falls_back_to_exact_when_taxonomy_has_nothing():
    out = LLMOutput(intent="FILTER_CANDIDATES", tools=[
        ToolItem(raw_text="totally-unknown-xyz-term", match_mode="expand", hard=True),
    ])
    filters, _ = resolve_filters(out, "knows totally-unknown-xyz-term")
    assert len(filters) == 1
    assert filters[0].operator == "contains"  # not "in" -- no list to widen into


def test_domain_hint_becomes_a_skip_note_not_a_filter():
    out = LLMOutput(intent="FILTER_CANDIDATES", domain_hint=["Finance", "ML/AI"])
    filters, skip_notes = resolve_filters(out, "finance folks")
    assert filters == []
    assert len(skip_notes) == 1
    assert "Finance" in skip_notes[0] and "ML/AI" in skip_notes[0]


def test_empty_output_resolves_to_nothing():
    filters, skip_notes = resolve_filters(LLMOutput(intent="FILTER_CANDIDATES"), "hello")
    assert filters == []
    assert skip_notes == []


def test_structured_and_tools_combine_in_one_call():
    out = LLMOutput(intent="FILTER_CANDIDATES",
                    structured=[StructuredItem(field="experience", operator="gte",
                                               raw_text="5 years", value=5)],
                    tools=[ToolItem(raw_text="Python", match_mode="exact")])
    filters, _ = resolve_filters(out, "5 years of Python experience")
    assert len(filters) == 2
    assert {f.field for f in filters} == {"experience", "skill"}


def test_item_not_mentioned_in_the_query_is_dropped_not_trusted():
    # Real, reported live bug: a session's SECOND turn ("who has worked on
    # supply chain platform") produced a fabricated
    # {"field":"skill_experience","skill":"ML","value":1} structured item --
    # zero mention of ML anywhere in that turn's query -- apparently bled in
    # from an earlier, unrelated turn ("give me a ML guy") despite this
    # turn's own replace_all=true asserting a standalone reinterpretation.
    # v2's raw_text/skill is supposed to be a literal SPAN of what was
    # actually said -- a term absent from the query violates that by
    # construction, so it must be dropped, not silently trusted.
    out = LLMOutput(intent="FILTER_CANDIDATES", replace_all=True,
                    structured=[StructuredItem(field="skill_experience", operator="gte",
                                               value=1, skill="ML", hard=True)],
                    tools=[ToolItem(raw_text="supply chain platform", match_mode="expand", hard=True)])
    filters, skip_notes = resolve_filters(out, "give me a guy who has worked on supply chain platform")
    assert [f.field for f in filters] == ["skill"]  # only the tool item survives
    assert any("ML" in note for note in skip_notes)


def test_short_term_grounding_uses_word_boundaries():
    # "ML" must not survive purely because some unrelated word in the query
    # happens to contain the letters "ml" as a substring (e.g. "html").
    out = LLMOutput(intent="FILTER_CANDIDATES",
                    structured=[StructuredItem(field="skill_experience", operator="gte",
                                               value=1, skill="ML", hard=True)])
    filters, skip_notes = resolve_filters(out, "knows html and css")
    assert filters == []
    assert skip_notes
    # But a real, word-boundary mention of "ML" must still be trusted.
    filters2, skip_notes2 = resolve_filters(out, "give me a ML guy")
    assert len(filters2) == 1
    assert skip_notes2 == []


def test_a_redundant_restatement_of_an_already_active_filter_stays_silent():
    # Real, reported live bug: a follow-up that ADDS to an existing filter
    # ("also needs AWS experience", after an earlier turn set Python)
    # caused the model to redundantly restate "Python" in its own raw
    # output for THIS turn -- normal multi-turn behavior (the recruiter's
    # "also" implicitly refers to what's already active), not a
    # hallucination. Since "python" isn't literally in this turn's text,
    # it gets dropped exactly like a genuine hallucination would -- but
    # merge_filters (using the SAME already-active state active_terms is
    # built from) re-adds it from session state regardless, so nothing is
    # actually lost, and the confusing "...carried over from an earlier,
    # UNRELATED question" note (factually wrong here -- Python is exactly
    # what this turn is building on) must not fire.
    out = LLMOutput(intent="FILTER_CANDIDATES",
                    tools=[ToolItem(raw_text="Python", match_mode="exact", hard=True),
                           ToolItem(raw_text="AWS", match_mode="exact", hard=True)])
    active = active_filter_terms([Filter(field="skill", operator="contains", value="Python")])
    filters, skip_notes = resolve_filters(out, "also needs AWS experience", active)
    assert {f.value for f in filters} == {"AWS"}  # Python dropped here...
    assert skip_notes == []  # ...but silently, since it's already active, not lost

    # The genuine hallucination case must still warn: an ungrounded term
    # that is NEITHER in this turn's text NOR already active.
    out2 = LLMOutput(intent="FILTER_CANDIDATES",
                     tools=[ToolItem(raw_text="Java", match_mode="exact", hard=True),
                            ToolItem(raw_text="AWS", match_mode="exact", hard=True)])
    filters2, skip_notes2 = resolve_filters(out2, "also needs AWS experience", active)
    assert {f.value for f in filters2} == {"AWS"}
    assert any("Java" in note for note in skip_notes2)


def test_item_with_no_raw_text_falls_back_to_grounding_on_its_own_value():
    # Real, reported live bug: with real conversation history present,
    # "candidate worked in high tire company" (a garbled "high tier
    # company" typo) resolved to a `company_type` item with raw_text=null
    # and value=["Product","Both"] -- ZERO connection to the actual query
    # text. Previously this had no grounding check at all (raw_text was
    # the only signal checked, and company_type isn't skill_experience/
    # domain_experience, so the `.skill` fallback doesn't apply either) --
    # it must now fall back to checking the item's own value(s).
    out = LLMOutput(intent="FILTER_CANDIDATES",
                    structured=[StructuredItem(field="company_type", operator="in",
                                               value=["Product", "Both"], hard=True)])
    filters, skip_notes = resolve_filters(out, "candidate worked in high tire company")
    assert filters == []
    assert any("Product" in note for note in skip_notes)

    # A genuinely grounded company_type item (the value really does appear
    # in the query) must still survive.
    filters2, skip_notes2 = resolve_filters(out, "product-based company, not services")
    assert len(filters2) == 1
    assert skip_notes2 == []


def test_no_raw_text_fallback_skips_numeric_and_boolean_fields():
    # A numeric value is often derived/paraphrased ("five years" -> 5, a
    # unit-converted notice_period) and a boolean value ("True"/"False")
    # never appears in natural-language text at all -- checking either
    # literally against the query would reject legitimate filters, not
    # catch hallucinated ones, so neither gets the value-fallback grounding
    # (see taxonomy._GROUNDABLE_VALUE_FIELD_TYPES).
    out = LLMOutput(intent="FILTER_CANDIDATES",
                    structured=[
                        StructuredItem(field="experience", operator="gte", value=5, hard=True),
                        StructuredItem(field="relocation", operator="equals", value=True, hard=True),
                    ])
    filters, skip_notes = resolve_filters(out, "someone with five years of experience, open to relocating")
    assert {f.field for f in filters} == {"experience", "relocation"}
    assert skip_notes == []
