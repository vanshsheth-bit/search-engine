"""Deterministic tests — no LLM required. Cover engine + validation + merge."""
from __future__ import annotations

import os

import pytest

from app.core.engine import apply_spec, matches_filter
from app.core.merge import merge_filters
from app.core.validation import validate_filters
from app.models.schemas import Filter, FilterSpec

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_requires_taxonomy = pytest.mark.skipif(
    not os.path.isfile(os.path.join(_ROOT, "merged_tools.json")),
    reason="merged_tools.json not present in this checkout",
)

CANDIDATES = [
    {"id": "c1", "name": "A", "match_score": 92, "location": "Mumbai",
     "experience": 6, "education": "Master", "notice_period_days": 15,
     "relocation": True,
     "skills": {"Python": {"years": 5}, "React": {"years": 2}}},
    {"id": "c2", "name": "B", "match_score": 87, "location": "Delhi",
     "experience": 4, "education": "Bachelor", "notice_period_days": 60,
     "skills": {"Python": {"years": 1}}},
    {"id": "c3", "name": "C", "match_score": 81, "location": "Mumbai",
     "experience": 8, "education": "Master", "notice_period_days": 0,
     "skills": {"Python": {"years": 6}, "Kubernetes": {"years": 3}}},
]


def _spec(*filters, logic="AND"):
    return FilterSpec(logic=logic, filters=list(filters))


def test_location_equals():
    spec = _spec(Filter(field="location", operator="equals", value="Mumbai"))
    out = apply_spec(CANDIDATES, spec)
    assert [c["id"] for c in out] == ["c1", "c3"]
    # scores preserved
    assert out[0]["match_score"] == 92


def test_skill_experience_gte():
    spec = _spec(Filter(field="skill_experience", operator="gte",
                        skill="Python", value=5))
    out = apply_spec(CANDIDATES, spec)
    assert {c["id"] for c in out} == {"c1", "c3"}


def test_combined_and():
    spec = _spec(
        Filter(field="location", operator="equals", value="Mumbai"),
        Filter(field="skill_experience", operator="gte", skill="Python", value=6),
    )
    out = apply_spec(CANDIDATES, spec)
    assert [c["id"] for c in out] == ["c3"]


def test_or_logic():
    spec = _spec(
        Filter(field="location", operator="equals", value="Delhi"),
        Filter(field="location", operator="equals", value="Mumbai"),
        logic="OR",
    )
    out = apply_spec(CANDIDATES, spec)
    assert {c["id"] for c in out} == {"c1", "c2", "c3"}


def test_not_contains_missing_skill_passes():
    # c2 has no Kubernetes -> "not_contains Kubernetes" should keep c2
    f = Filter(field="skill", operator="not_contains", value="Kubernetes")
    assert matches_filter(CANDIDATES[1], f) is True
    assert matches_filter(CANDIDATES[2], f) is False  # c3 has Kubernetes


def test_notice_period_lte():
    spec = _spec(Filter(field="notice_period", operator="lte", value=30,
                        unit="days"))
    out = apply_spec(CANDIDATES, spec)
    assert {c["id"] for c in out} == {"c1", "c3"}


def test_education_ordinal():
    spec = _spec(Filter(field="education", operator="gte", value="Master"))
    out = apply_spec(CANDIDATES, spec)
    assert {c["id"] for c in out} == {"c1", "c3"}


def test_education_equals_is_phrasing_robust():
    # "Masters" (plural, as an LLM might phrase it) must still match
    # candidates whose stored value is the canonical singular "Master".
    for phrasing in ("Masters", "Master's", "MS", "master"):
        spec = _spec(Filter(field="education", operator="equals", value=phrasing))
        out = apply_spec(CANDIDATES, spec)
        assert {c["id"] for c in out} == {"c1", "c3"}, phrasing


def test_contains_on_list_field_is_substring_not_exact():
    # university (like data from candidates.py) is a list of full free-text
    # names, e.g. ["KJ Somaiya School of Engineering, Mumbai, India"]. A
    # "contains" query for "Somaiya" must match via substring, not require
    # the whole list entry to equal "Somaiya" exactly.
    cand = {"id": "x", "university": ["KJ Somaiya School of Engineering, Mumbai, India"]}
    f = Filter(field="university", operator="contains", value="Somaiya")
    assert matches_filter(cand, f) is True


def test_skill_contains_is_exact_not_substring():
    # skills are atomic tokens, unlike university names -- "java" must NOT
    # match a candidate who only has "javascript".
    cand = {"id": "x", "skills": ["javascript", "html"]}
    f = Filter(field="skill", operator="contains", value="java")
    assert matches_filter(cand, f) is False

    cand2 = {"id": "y", "skills": ["java", "html"]}
    assert matches_filter(cand2, f) is True


def test_education_rank_no_false_positive_substring():
    from app.core.vocabulary import education_rank
    # "Systems" contains the substring "ms" -- must not be mistaken for "MS".
    assert education_rank("Bachelor of Science in Information Systems") == 3
    assert education_rank("Diploma in Computer Systems") == 2


def test_education_gte_passes_validation():
    # education is an "ordinal" field type -- gte must be a legal operator
    # for it, not just for numeric fields.
    res = validate_filters(
        [Filter(field="education", operator="gte", value="Master")]
    )
    assert res.ok is True


def test_scores_never_recalculated():
    spec = _spec(Filter(field="location", operator="equals", value="Mumbai"))
    out = apply_spec(CANDIDATES, spec)
    assert out[0]["match_score"] == 92
    assert out[1]["match_score"] == 81


def test_merge_replaces_location():
    existing = [Filter(field="location", operator="equals", value="Mumbai")]
    incoming = [Filter(field="location", operator="equals", value="Bangalore")]
    merged = merge_filters(existing, incoming)
    assert len(merged) == 1
    assert merged[0].value == "Bangalore"


def test_merge_keeps_distinct_skills():
    existing = [Filter(field="skill_experience", operator="gte",
                       skill="Python", value=3)]
    incoming = [Filter(field="skill_experience", operator="gte",
                       skill="React", value=2)]
    merged = merge_filters(existing, incoming)
    assert len(merged) == 2


def test_merge_keeps_two_plain_skill_filters_named_together():
    # "someone with experience in react and python" -> the LLM emits two
    # plain "skill" filters (not skill_experience) in the SAME batch. Both
    # have skill=None (that attribute only carries a value for
    # skill_experience), so a naive key() would collide them onto the same
    # key and silently drop one -- must not happen.
    incoming = [
        Filter(field="skill", operator="contains", value="React"),
        Filter(field="skill", operator="contains", value="Python"),
    ]
    merged = merge_filters([], incoming)
    assert len(merged) == 2
    values = {f.value for f in merged}
    assert values == {"React", "Python"}


def test_merge_still_replaces_single_value_fields():
    # Non-multi-value fields (location, education, ...) keep "replace"
    # semantics -- unlike skill/university/company, there's only ever one
    # sensible current answer.
    existing = [Filter(field="education", operator="gte", value="Bachelor")]
    incoming = [Filter(field="education", operator="gte", value="Master")]
    merged = merge_filters(existing, incoming)
    assert len(merged) == 1
    assert merged[0].value == "Master"


def test_validation_rejects_unknown_field():
    res = validate_filters([Filter(field="salary", operator="gte", value=10)])
    assert res.ok is False
    assert res.unsupported is True


def test_validation_rejects_bad_operator_for_type():
    res = validate_filters(
        [Filter(field="experience", operator="contains", value="x")]
    )
    assert res.ok is False


def test_validation_coerces_numeric_string():
    res = validate_filters(
        [Filter(field="experience", operator="gte", value="3")]
    )
    assert res.ok is True
    assert res.filters[0].value == 3


def test_validation_requires_skill_for_skill_experience():
    res = validate_filters(
        [Filter(field="skill_experience", operator="gte", value=3)]
    )
    assert res.ok is False


def test_unavailable_field_in_dataset():
    res = validate_filters(
        [Filter(field="relocation", operator="equals", value=True)],
        available_fields={"location", "skill"},
    )
    assert res.ok is False
    assert res.unsupported is True


def test_generic_word_rejected_as_university_name():
    # A weak model can turn "good universities" into a literal name search
    # for "good" -- must be caught, not silently applied as a real filter.
    res = validate_filters(
        [Filter(field="university", operator="contains", value="good")]
    )
    assert res.ok is False
    assert res.unsupported is False  # this is a bad value, not a missing field
    assert "good" in res.error


def test_real_university_name_still_passes():
    res = validate_filters(
        [Filter(field="university", operator="contains", value="Somaiya")]
    )
    assert res.ok is True


def test_compound_query_degrades_gracefully_instead_of_failing_whole_request():
    # "8+ years, knows Kubernetes, and open to relocating" -- relocation isn't
    # available for this dataset, but experience/skill are real and valid.
    # The bad clause must not sink the two good ones.
    res = validate_filters(
        [
            Filter(field="experience", operator="gte", value=8),
            Filter(field="skill", operator="contains", value="Kubernetes"),
            Filter(field="relocation", operator="equals", value=True),
        ],
        available_fields={"location", "experience", "skill"},
    )
    assert res.ok is True
    assert {f.field for f in res.filters} == {"experience", "skill"}
    assert any("relocate" in s.lower() for s in res.skipped)


def test_in_operator_on_list_valued_field_checks_membership_not_stringified_list():
    # Regression: "in" on a list-valued candidate field (skill) must check
    # whether ANY of the candidate's items matches ANY of the filter's
    # values -- not stringify the whole candidate list and compare it as one
    # blob against each value (which would essentially never match).
    cand = {"id": "x", "skills": ["python", "react"]}
    f = Filter(field="skill", operator="in", value=["machine learning", "python", "aws"])
    assert matches_filter(cand, f) is True

    cand2 = {"id": "y", "skills": ["java", "go"]}
    assert matches_filter(cand2, f) is False


def test_not_in_operator_on_list_valued_field():
    cand = {"id": "x", "skills": ["kubernetes"]}
    f = Filter(field="skill", operator="not_in", value=["docker", "kubernetes"])
    assert matches_filter(cand, f) is False

    cand2 = {"id": "y", "skills": ["terraform"]}
    assert matches_filter(cand2, f) is True


# --------------------------------------------------------------------------- #
# "<Skill> developer" job_title self-heal (validation.py) -- regression for a
# confirmed live bug: "python developer" was routed as a literal job_title
# phrase that matches 0 of 103 real candidates, when 40 of them really have
# Python as a declared skill. See validation.py's _GENERIC_DEV_SUFFIXES
# docstring for the full rationale.
# --------------------------------------------------------------------------- #
@_requires_taxonomy
def test_skill_developer_phrase_heals_to_skill_filter():
    res = validate_filters(
        [Filter(field="job_title", operator="contains", value="Python developer")]
    )
    assert res.ok is True
    assert len(res.filters) == 1
    assert res.filters[0].field == "skill"
    assert res.filters[0].operator == "contains"
    assert res.filters[0].value == "Python"


@_requires_taxonomy
def test_skill_developer_phrase_heal_preserves_negation():
    res = validate_filters(
        [Filter(field="job_title", operator="not_contains", value="Java developer")]
    )
    assert res.ok is True
    assert res.filters[0].field == "skill"
    assert res.filters[0].operator == "not_contains"
    assert res.filters[0].value == "Java"


@_requires_taxonomy
def test_skill_developer_phrase_heal_drops_leading_modifier():
    # "Senior Python Developer" -- the full prefix "senior python" isn't a
    # real tool, but the word immediately before "developer" is.
    res = validate_filters(
        [Filter(field="job_title", operator="contains", value="Senior Python Developer")]
    )
    assert res.ok is True
    assert res.filters[0].field == "skill"
    assert res.filters[0].value == "Python"


@_requires_taxonomy
def test_generic_dev_suffix_alone_is_not_healed():
    # "Software Developer" is a real, common title -- "software" is not a
    # taxonomy tool, so this must stay job_title untouched.
    res = validate_filters(
        [Filter(field="job_title", operator="contains", value="Software Developer")]
    )
    assert res.ok is True
    assert res.filters[0].field == "job_title"
    assert res.filters[0].value == "Software Developer"


def test_engineer_suffix_is_never_healed():
    # Deliberately out of scope even without needing the taxonomy at all --
    # "engineer" titles (DevOps Engineer, ML Engineer, Data Engineer, ...)
    # are established standalone conventions, not a generic-role-noun
    # standing in for a skill. Must survive completely unchanged.
    res = validate_filters(
        [Filter(field="job_title", operator="contains", value="DevOps Engineer")]
    )
    assert res.ok is True
    assert res.filters[0].field == "job_title"
    assert res.filters[0].value == "DevOps Engineer"


# --------------------------------------------------------------------------- #
# Country-abbreviation-in-"location" self-heal (validation.py) -- regression
# for a confirmed live eval failure: "engineers based in the UAE" routed
# "UAE" into "location" (a city field) instead of "country", which can never
# match any real candidate.
# --------------------------------------------------------------------------- #
def test_location_country_abbreviation_heals_to_country_filter():
    res = validate_filters(
        [Filter(field="location", operator="equals", value="UAE")]
    )
    assert res.ok is True
    assert res.filters[0].field == "country"
    assert res.filters[0].value == "United Arab Emirates"


def test_location_real_city_name_is_never_healed():
    # A real city must never be reinterpreted as a country -- only the
    # small, known set of country abbreviations triggers this heal.
    res = validate_filters(
        [Filter(field="location", operator="equals", value="Mumbai")]
    )
    assert res.ok is True
    assert res.filters[0].field == "location"
    assert res.filters[0].value == "Mumbai"


# --------------------------------------------------------------------------- #
# notice_period unit default (validation.py) -- regression for a confirmed
# live eval failure where the model omitted "unit" on an otherwise-correct
# notice_period filter.
# --------------------------------------------------------------------------- #
def test_notice_period_missing_unit_defaults_to_days():
    res = validate_filters(
        [Filter(field="notice_period", operator="lte", value=90)]
    )
    assert res.ok is True
    assert res.filters[0].unit == "days"


def test_notice_period_explicit_unit_is_not_overridden():
    res = validate_filters(
        [Filter(field="notice_period", operator="lte", value=3, unit="months")]
    )
    assert res.ok is True
    assert res.filters[0].unit == "months"


# --------------------------------------------------------------------------- #
# Generic skill-filler-word guard (validation.py) -- regression for a
# confirmed live bug: the bare query "Skills" (naming no real technology)
# was parsed as a literal skill value, then the fuzzy-matching pipeline
# treated the meaningless term as real and "verified" candidates against
# it. Same class of guard as the existing university/company one below.
# --------------------------------------------------------------------------- #
def test_generic_skill_word_rejected():
    res = validate_filters(
        [Filter(field="skill", operator="contains", value="Skills")]
    )
    assert res.ok is False
    assert "Skills" in res.error


def test_generic_skill_words_all_rejected():
    for word in ("skill", "experience", "expertise", "knowledge", "technology",
                 "tools", "ability", "qualifications"):
        res = validate_filters(
            [Filter(field="skill", operator="contains", value=word)]
        )
        assert res.ok is False, word


def test_real_skill_name_still_passes():
    res = validate_filters(
        [Filter(field="skill", operator="contains", value="Python")]
    )
    assert res.ok is True
    assert res.filters[0].value == "Python"


def test_generic_skill_word_does_not_sink_a_compound_query():
    # A bad "skill" clause must not abort an otherwise-valid compound
    # query -- same graceful-degradation principle as the existing
    # relocation/Kubernetes test above.
    res = validate_filters(
        [
            Filter(field="experience", operator="gte", value=5),
            Filter(field="skill", operator="contains", value="Skills"),
        ]
    )
    assert res.ok is True
    assert {f.field for f in res.filters} == {"experience"}
    assert any("Skills" in s for s in res.skipped)
