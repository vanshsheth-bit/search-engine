"""Deterministic tests — no LLM required. Cover engine + validation + merge."""
from __future__ import annotations

from app.core.engine import apply_spec, matches_filter
from app.core.merge import merge_alternative_groups, merge_filters, to_chips
from app.core.validation import validate_alternative_groups, validate_filters
from app.core.vocabulary import seniority_band
from app.models.schemas import AlternativeGroup, Filter, FilterSpec

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


def test_domain_experience_gte_sums_per_experience_not_total_career():
    # Exact reported scenario: 4 real jobs (3y SDE, 2y DevOps, 4y SDE, 1y
    # SDE) -- someone has 8 years of SDE experience and 2 years of DevOps,
    # NOT their 10-year total career length for either (see
    # candidates._load_candidate_domain_years, which computes domain_years
    # this way from real per-experience durations).
    candidates = [
        {"id": "d1", "name": "D1", "experience": 10,
         "domain_years": {"Software Engineering": 8.0, "DevOps": 2.0}},
        {"id": "d2", "name": "D2", "experience": 10,
         "domain_years": {"Software Engineering": 9.0, "DevOps": 1.0}},
        {"id": "d3", "name": "D3", "experience": 3,
         "domain_years": {"DevOps": 3.0}},
    ]
    spec = _spec(Filter(field="domain_experience", operator="gte",
                        skill="DevOps", value=2))
    out = apply_spec(candidates, spec)
    # d1 (2.0 DevOps) and d3 (3.0 DevOps) qualify; d2 (1.0 DevOps, despite
    # 10 years total career and 9 years of SDE) correctly does not.
    assert {c["id"] for c in out} == {"d1", "d3"}


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


def test_merge_alternative_groups_wholesale_replace_not_per_key():
    # Unlike merge_filters (per-key dedup), a stated "either route" is one
    # coherent clause -- there's no sensible leaf-by-leaf merge across two
    # turns' two DIFFERENT routes, so a non-empty new statement fully
    # replaces the old one.
    old = [AlternativeGroup(filters=[Filter(field="experience", operator="gte", value=10)])]
    new = [AlternativeGroup(filters=[Filter(field="education", operator="gte", value="Master")])]
    assert merge_alternative_groups(old, new, replace_all=False) == new


def test_merge_alternative_groups_carries_over_when_not_mentioned():
    # A refinement turn that doesn't mention alternative_groups at all keeps
    # whatever was active -- same continuity convention as an unmentioned
    # flat filter surviving a refinement query.
    old = [AlternativeGroup(filters=[Filter(field="experience", operator="gte", value=10)])]
    assert merge_alternative_groups(old, [], replace_all=False) == old


def test_merge_alternative_groups_replace_all_clears_even_when_incoming_empty():
    old = [AlternativeGroup(filters=[Filter(field="experience", operator="gte", value=10)])]
    assert merge_alternative_groups(old, [], replace_all=True) == []


def test_to_chips_alternative_groups_produce_exactly_one_synthetic_chip():
    # Never one chip per group -- that would misrepresent "any ONE of these
    # routes is required" as a set of unrelated separate facts.
    chips = to_chips(
        [Filter(field="location", operator="equals", value="Mumbai")],
        [
            AlternativeGroup(filters=[
                Filter(field="education", operator="gte", value="Master"),
                Filter(field="college_tier", operator="gte", value="High"),
            ]),
            AlternativeGroup(filters=[Filter(field="experience", operator="gte", value=10)]),
        ],
    )
    assert len(chips) == 2
    assert chips[0].field == "location"
    group_chip = chips[1]
    assert group_chip.field == "_alternative_group"
    assert "OR" in group_chip.label
    assert "AND" in group_chip.label  # the Master's+High-tier route is one AND'd clause


def test_to_chips_no_group_chip_when_alternative_groups_absent():
    chips = to_chips([Filter(field="location", operator="equals", value="Mumbai")])
    assert len(chips) == 1
    assert all(c.field != "_alternative_group" for c in chips)


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


def test_validation_requires_skill_for_domain_experience():
    # domain_experience reuses `skill` generically as "the named target this
    # number refers to" (a domain/subdomain name here, not a skill) -- see
    # vocabulary.SKILL_SCOPED_FIELDS.
    res = validate_filters(
        [Filter(field="domain_experience", operator="gte", value=2)]
    )
    assert res.ok is False


def test_domain_experience_availability_probes_domain_field():
    # domain_experience's data-availability check maps to "domain" (the
    # same underlying classifier data), not a separate "domain_experience"
    # key -- see validation.py's probe logic.
    res = validate_filters(
        [Filter(field="domain_experience", operator="gte", skill="DevOps", value=2)],
        available_fields={"location", "skill"},  # no "domain"
    )
    assert res.ok is False
    assert res.unsupported is True

    res2 = validate_filters(
        [Filter(field="domain_experience", operator="gte", skill="DevOps", value=2)],
        available_fields={"location", "skill", "domain"},
    )
    assert res2.ok is True


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
# alternative_groups -- see schemas.AlternativeGroup and
# engine.apply_spec's docstring for why this exists: "require EITHER a
# Master's from a Tier-1 university OR 10+ years of experience" is a real
# cross-field OR-of-AND-groups requirement that spec.logic (one flat
# operator over the whole filters list) cannot express alongside other
# AND'd requirements in the same query.
# --------------------------------------------------------------------------- #
_ROUTE_CANDIDATES = [
    {"id": "r1", "name": "MastersHighTier", "match_score": 90,
     "education": "Master", "college_tier": "High", "experience": 3},
    {"id": "r2", "name": "TenYearsNoMasters", "match_score": 85,
     "education": "Bachelor", "college_tier": "Low", "experience": 12},
    {"id": "r3", "name": "NeitherRoute", "match_score": 80,
     "education": "Bachelor", "college_tier": "Low", "experience": 4},
    {"id": "r4", "name": "MastersLowTier", "match_score": 75,
     "education": "Master", "college_tier": "Low", "experience": 2},
]


def _masters_or_ten_years_spec(**kwargs):
    return FilterSpec(
        logic="AND", filters=[], alternative_groups=[
            AlternativeGroup(filters=[
                Filter(field="education", operator="gte", value="Master"),
                Filter(field="college_tier", operator="gte", value="High"),
            ]),
            AlternativeGroup(filters=[
                Filter(field="experience", operator="gte", value=10),
            ]),
        ],
        **kwargs,
    )


def test_alternative_groups_or_gate_admits_either_route():
    out = apply_spec(_ROUTE_CANDIDATES, _masters_or_ten_years_spec())
    names = {c["name"] for c in out}
    # r1 satisfies route 1 (Master's + High tier). r2 satisfies route 2
    # (10+ years) despite having NEITHER a Master's NOR a high-tier college
    # -- proving this is a real OR, not an accidental AND of both routes.
    assert names == {"MastersHighTier", "TenYearsNoMasters"}
    # r4 has a Master's but NOT a high-tier college (route 1 needs BOTH,
    # AND'd) and only 2 years (route 2 needs 10+) -- satisfies neither
    # route in full, correctly excluded despite partially matching route 1.
    assert "MastersLowTier" not in names
    assert "NeitherRoute" not in names


def test_alternative_groups_combine_with_flat_and_filters():
    # A candidate must ALSO satisfy the ordinary flat AND filters -- the
    # group gate is an ADDITIONAL requirement, not a replacement for logic/
    # filters. r2 (route 2: 10+ years) would pass the group gate alone, but
    # adding a flat requirement it fails (education gte Master) excludes it.
    spec = _masters_or_ten_years_spec()
    spec = spec.model_copy(update={
        "filters": [Filter(field="education", operator="gte", value="Master")],
    })
    out = apply_spec(_ROUTE_CANDIDATES, spec)
    names = {c["name"] for c in out}
    # r1: Master's (flat AND) + Master's/High tier (route 1) -- passes both.
    assert names == {"MastersHighTier"}


def test_alternative_groups_empty_group_never_vacuously_passes():
    # A group with an empty filters list must never vacuously satisfy the
    # OR gate (all([]) is True in Python) and defeat the whole requirement
    # -- confirmed via the explicit bool(g.filters) guard in apply_spec.
    spec = FilterSpec(logic="AND", filters=[], alternative_groups=[
        AlternativeGroup(filters=[]),
        AlternativeGroup(filters=[Filter(field="experience", operator="gte", value=999)]),
    ])
    out = apply_spec(_ROUTE_CANDIDATES, spec)
    assert out == []


def test_alternative_groups_no_op_when_absent():
    # A spec with no alternative_groups at all behaves exactly as before --
    # this is the "did I break the existing flat-only case" regression check.
    spec = FilterSpec(logic="AND", filters=[Filter(field="experience", operator="gte", value=3)])
    out = apply_spec(_ROUTE_CANDIDATES, spec)
    names = {c["name"] for c in out}
    assert names == {"MastersHighTier", "TenYearsNoMasters", "NeitherRoute"}


def test_validate_alternative_groups_drops_whole_group_on_invalid_leaf():
    # If ANY leaf in a group fails validation, the ENTIRE group is dropped
    # -- not silently weakened to just the leaves that happened to be valid
    # (see validate_alternative_groups' docstring for why that's worse).
    groups = [
        AlternativeGroup(filters=[
            Filter(field="education", operator="gte", value="Master"),
            Filter(field="not_a_real_field", operator="equals", value="x"),
        ]),
        AlternativeGroup(filters=[Filter(field="experience", operator="gte", value=10)]),
    ]
    validated, notes = validate_alternative_groups(groups)
    assert len(validated) == 1
    assert validated[0].filters == [Filter(field="experience", operator="gte", value=10, hard=True)]
    assert notes  # explains the dropped route


def test_validate_alternative_groups_forces_hard_true():
    # A soft preference has no meaning inside an eligibility route -- every
    # surviving leaf is forced hard=True regardless of what was set.
    groups = [AlternativeGroup(filters=[
        Filter(field="experience", operator="gte", value=10, hard=False),
    ])]
    validated, _ = validate_alternative_groups(groups)
    assert validated[0].filters[0].hard is True


def test_validate_alternative_groups_drops_all_when_every_group_invalid():
    groups = [AlternativeGroup(filters=[
        Filter(field="not_a_real_field", operator="equals", value="x"),
    ])]
    validated, notes = validate_alternative_groups(groups)
    assert validated == []
    assert notes


# --------------------------------------------------------------------------- #
# Seniority-band canonicalization (vocabulary.seniority_band) and the
# resulting OR-of-routes semantics at the engine level.
# --------------------------------------------------------------------------- #
def test_seniority_band_recognizes_every_canonical_band_and_alias():
    assert seniority_band("fresher") == "fresher"
    assert seniority_band("Entry Level") == "fresher"
    assert seniority_band("junior") == "junior"
    assert seniority_band("Jr") == "junior"
    assert seniority_band("mid level") == "mid"
    assert seniority_band("Mid-Level") == "mid"
    assert seniority_band("intermediate") == "mid"
    assert seniority_band("senior") == "senior"
    assert seniority_band("Sr") == "senior"
    assert seniority_band("lead") == "lead"
    assert seniority_band("principal") == "lead"
    assert seniority_band("staff") == "lead"


def test_seniority_band_returns_none_for_unrecognized_or_empty():
    assert seniority_band("experienced") is None
    assert seniority_band("blah level") is None
    assert seniority_band(None) is None
    assert seniority_band("") is None


_SENIORITY_CANDIDATES = [
    {"id": "s1", "name": "YearsOnlyMatch", "match_score": 90,
     "experience": 5, "job_title": ["Backend Engineer"]},
    {"id": "s2", "name": "TitleOnlyMatch", "match_score": 85,
     "experience": 15, "job_title": ["Mid-Level Software Engineer"]},
    {"id": "s3", "name": "NeitherMatch", "match_score": 80,
     "experience": 15, "job_title": ["Backend Engineer"]},
]


def _mid_band_spec():
    # Mirrors exactly what service._expand_seniority_filters builds for a
    # HARD "mid level" query: one route on the years range, one route on
    # the job_title keyword check, OR'd against each other.
    return FilterSpec(logic="AND", filters=[], alternative_groups=[
        AlternativeGroup(filters=[
            Filter(field="experience", operator="gte", value=3),
            Filter(field="experience", operator="lte", value=7),
        ]),
        AlternativeGroup(filters=[
            Filter(field="job_title", operator="in", value=["Mid-Level", "Mid Level", "Intermediate Engineer"]),
        ]),
    ])


def test_seniority_band_or_gate_admits_either_years_or_title_signal():
    out = apply_spec(_SENIORITY_CANDIDATES, _mid_band_spec())
    names = {c["name"] for c in out}
    # s1 satisfies via years alone (5, in [3,7]) despite a title with no
    # level word. s2 satisfies via title alone (15 years is well outside the
    # band) -- proving this is a real EITHER-signal OR, not an accidental AND.
    assert names == {"YearsOnlyMatch", "TitleOnlyMatch"}
    assert "NeitherMatch" not in names
