"""Unit tests for pure helper functions in app.core.candidates that don't
need the real or synthetic dataset loaded, plus real-data tests for the
`domain` field (needs the real dataset + a built experience_index -- see
scripts/build_experience_index.py; skipped automatically if that hasn't
been built on this machine, same graceful-degradation the app itself has)."""
from __future__ import annotations

import pytest

from app.core.candidates import (
    _skill_years_from_experience,
    canonicalize_country,
    get_matched_candidates,
)
from app.core.engine import apply_spec
from app.core.experience_index import IndexPaths
from app.models.schemas import Filter, FilterSpec

# The ML-diverse real job used throughout this session's manual testing --
# see app/core/candidates.py's module docstring and search-ui/index.html's
# JOB_ID comment for why this one (not the location-diverse one) is used
# for skill/domain-style real-data tests.
_JOB = "00000103"

_index_not_built = pytest.mark.skipif(
    not IndexPaths().exists(),
    reason="experience_index not built on this machine -- run "
           "scripts/build_experience_index.py first",
)


def test_canonicalize_country_resolves_common_aliases():
    assert canonicalize_country("USA") == "United States"
    assert canonicalize_country("usa") == "United States"
    assert canonicalize_country("US") == "United States"
    assert canonicalize_country("America") == "United States"
    assert canonicalize_country("UK") == "United Kingdom"
    assert canonicalize_country("Britain") == "United Kingdom"
    assert canonicalize_country("UAE") == "United Arab Emirates"


def test_canonicalize_country_leaves_unknown_names_unchanged():
    # Not in the alias table -- the LLM's own full-name guess is trusted
    # as-is, this only corrects known colloquial short forms.
    assert canonicalize_country("Japan") == "Japan"
    assert canonicalize_country("India") == "India"


def test_canonicalize_country_handles_none_and_empty():
    assert canonicalize_country(None) is None
    assert canonicalize_country("") == ""


def test_skill_years_sums_duration_across_every_job_that_names_the_skill():
    experience = [
        {"position": "Backend Engineer", "description": "Built services in Python and Go.",
         "duration_years": 2.0},
        {"position": "Senior Engineer", "description": "Led a Python team.", "duration_years": 1.5},
        {"position": "Support Analyst", "description": "Handled tickets.", "duration_years": 3.0},
    ]
    out = _skill_years_from_experience(["Python"], experience)
    assert out == {"Python": {"years": 3.5}}  # 2.0 + 1.5, third job never mentions it


def test_skill_years_none_for_a_skill_never_named_in_any_job_text():
    experience = [{"position": "Engineer", "description": "Did engineering things.",
                   "duration_years": 4.0}]
    out = _skill_years_from_experience(["Python"], experience)
    assert out == {"Python": {"years": None}}


def test_skill_years_none_for_an_unrecognized_skill_name():
    # Not a known tool at all (see skill_taxonomy.is_known_tool) -- never
    # attempts a text search, regardless of what the description says.
    experience = [{"position": "Engineer", "description": "Focused on route safety.",
                   "duration_years": 4.0}]
    out = _skill_years_from_experience(["route", "safety"], experience)
    assert out == {"route": {"years": None}, "safety": {"years": None}}


def test_skill_years_excludes_known_tools_that_are_also_common_english_words():
    # "Excel"/"Go" ARE real, recognized taxonomy tools (unlike "route"/
    # "safety" above) -- but a dataset-wide audit found the word is used
    # almost exclusively in its generic English sense in this corpus's
    # prose (see _AMBIGUOUS_FOR_YEARS_TEXT_MATCH's comment), so a mention
    # must NOT be trusted as evidence of real tool experience even though
    # the literal word is right there in the text.
    experience = [{"position": "Manager", "description": "Worked hard to excel and go the extra mile.",
                   "duration_years": 5.0}]
    out = _skill_years_from_experience(["Excel", "Go"], experience)
    assert out == {"Excel": {"years": None}, "Go": {"years": None}}


@_index_not_built
def test_domain_field_populates_from_real_classified_experience_data():
    cands = get_matched_candidates(_JOB)
    with_domain = [c for c in cands if c.get("domain")]
    # Real, verified count on this dataset -- most candidates have SOME
    # classified experience, not all (some have no description text to
    # classify at all -- see experience_text.embedding_text).
    assert len(with_domain) > 0
    for c in with_domain:
        assert isinstance(c["domain"], list)
        assert all(isinstance(d, str) and d for d in c["domain"])


@_index_not_built
def test_domain_contains_fintech_finds_real_verified_candidate():
    # Real, verified result on this dataset (confirmed manually before
    # writing this test): exactly one real candidate on this job has a
    # FinTech-related classified subdomain.
    cands = get_matched_candidates(_JOB)
    spec = FilterSpec(logic="AND", filters=[
        Filter(field="domain", operator="contains", value="fintech"),
    ])
    matched = apply_spec(cands, spec)
    names = {c["name"] for c in matched}
    assert "Adam N Schmidt" in names
    for c in matched:
        assert any("fintech" in d.lower() for d in c["domain"])
