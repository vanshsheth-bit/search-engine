"""Deterministic tests — no LLM required. Cover engine + validation + merge."""
from __future__ import annotations

from app.core.engine import apply_spec, matches_filter
from app.core.merge import merge_filters
from app.core.validation import validate_filters
from app.models.schemas import Filter, FilterSpec

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
