"""Unit tests for pure helper functions in app.core.candidates that don't
need the real or synthetic dataset loaded, plus real-data tests for the
`domain` field (needs the real dataset + a built experience_index -- see
scripts/build_experience_index.py; skipped automatically if that hasn't
been built on this machine, same graceful-degradation the app itself has)."""
from __future__ import annotations

import pytest

from app.core.candidates import canonicalize_country, get_matched_candidates, _extract_skill_years
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


def test_skill_years_credited_from_experience_description():
    experience = [
        {"description": "Technologies: Java, Python, Docker", "duration_years": 4.5},
    ]
    years = _extract_skill_years(experience, ["Python", "Java", "Docker", "AWS"])
    assert years == {"python": 4.5, "java": 4.5, "docker": 4.5}
    assert "aws" not in years  # never mentioned -- absent, not zero


def test_skill_years_sums_across_multiple_experiences():
    experience = [
        {"description": "Built services in Python and Go", "duration_years": 2.0},
        {"description": "Led a Python team using AWS Lambda", "duration_years": 3.0},
    ]
    years = _extract_skill_years(experience, ["Python", "Go", "AWS"])
    assert years["python"] == 5.0
    assert years["go"] == 2.0
    assert years["aws"] == 3.0


def test_skill_years_word_boundary_avoids_false_positives():
    # "R" and "Go" are real skill names but also common English words/
    # substrings -- a naive substring/short-token match would wrongly
    # credit them from "Report" and "ago"/"algorithm".
    experience = [
        {"description": "Wrote a quarterly Report and improved algorithm efficiency",
         "duration_years": 3.0},
    ]
    years = _extract_skill_years(experience, ["R", "Go"])
    assert years == {}


def test_skill_years_handles_symbol_suffixed_skill_names():
    experience = [
        {"description": "Maintained legacy services in C++ and C#", "duration_years": 6.0},
    ]
    years = _extract_skill_years(experience, ["C++", "C#", "C"])
    assert years["c++"] == 6.0
    assert years["c#"] == 6.0
    assert "c" not in years  # bare "C" never actually appears standalone


def test_skill_years_ignores_experiences_with_no_duration():
    experience = [{"description": "Worked with Python", "duration_years": 0}]
    assert _extract_skill_years(experience, ["Python"]) == {}


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
