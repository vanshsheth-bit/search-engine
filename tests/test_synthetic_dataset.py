"""Regression tests against the adversarial synthetic dataset in
test_data/ (see test_data/generate_synthetic_data.py for what each of the
25 candidates is specifically designed to probe).

Unlike testing against the real dataset -- where verifying a result meant
manually auditing candidate records by hand -- every property here is
authored, so the correct answer to each query below is KNOWN, not inferred.
This caught three real bugs during construction (all fixed, asserted here
so they can't silently regress):
  - "Doctor of Philosophy" wasn't recognized as any degree level at all
    (vocabulary.py's _DEGREE_KEYWORDS only had the abbreviation "PhD").
  - A freelancer's placeholder "company" ("Self-employed", "Startup
    (Confidential)") matched real junk rows in company_ranks.json via the
    prefix fallback, assigning a fake tier to a non-company.
  - Naming variants of the same tool ("react.js"/"ReactJS"/"React JS")
    didn't match a plain "React" query -- candidate skills are now
    canonicalized through the same taxonomy used for query expansion.

Isolated from the real-data tests via monkeypatching app.core.candidates'
module-level path constants and clearing its lru_caches around each test,
so this never leaks into (or is polluted by) tests using the real dataset.
"""
from __future__ import annotations

import os

import pytest

from app.core import candidates as C
from app.core.engine import apply_spec
from app.models.schemas import Filter, FilterSpec

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
_RESUMES = os.path.join(_TEST_DATA_DIR, "synthetic_parsedresumes.json")
_MATCHES = os.path.join(_TEST_DATA_DIR, "synthetic_jdmatchresults.json")

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(_RESUMES) and os.path.isfile(_MATCHES)),
    reason="synthetic test_data not present in this checkout",
)

JOB_BACKEND = "SYN-BACKEND-01"
JOB_ML = "SYN-ML-02"


@pytest.fixture(scope="module")
def synthetic():
    """Point app.core.candidates at the synthetic dataset for this whole
    test module (all tests here are read-only queries, so sharing one load
    is safe), then restore the real paths and clear every cache both ways
    so no state leaks into other test modules. Module-scoped deliberately:
    _load_company_tiers streams the real ~900MB company_ranks.json on a
    cache miss, so re-clearing it per-test would re-scan that file once per
    test instead of once total."""
    orig_resumes, orig_matches = C._PARSED_RESUMES_PATH, C._JD_MATCH_RESULTS_PATH
    C._PARSED_RESUMES_PATH, C._JD_MATCH_RESULTS_PATH = _RESUMES, _MATCHES
    for fn in (C._load_raw_resumes, C._load_resumes_by_process_id,
               C._load_matches_by_job, C._load_company_tiers,
               C._load_university_tiers, C._load_city_gazetteer):
        fn.cache_clear()
    yield
    C._PARSED_RESUMES_PATH, C._JD_MATCH_RESULTS_PATH = orig_resumes, orig_matches
    for fn in (C._load_raw_resumes, C._load_resumes_by_process_id,
               C._load_matches_by_job, C._load_company_tiers,
               C._load_university_tiers, C._load_city_gazetteer):
        fn.cache_clear()


def _by_name(candidates, name):
    matches = [c for c in candidates if c["name"] == name]
    assert matches, f"{name!r} not found among {[c['name'] for c in candidates]}"
    return matches[0]


def test_dataset_loads_without_error(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    assert len(backend) == 24  # 25 candidates minus 1 "failed" status


def test_failed_status_is_excluded(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    names = {c["name"] for c in backend}
    assert "Priyanka Yadav" not in names  # her only match record is status="failed"


def test_job_scoping_ml_pool_is_a_strict_subset(synthetic):
    backend = {c["id"] for c in C.get_matched_candidates(JOB_BACKEND)}
    ml = {c["id"] for c in C.get_matched_candidates(JOB_ML)}
    assert len(ml) == 6
    assert ml.issubset(backend)


# --------------------------------------------------------------------------- #
# Location: historical Indian city name aliasing + hyphenated compounds
# --------------------------------------------------------------------------- #
def test_bombay_and_mumbai_resolve_to_the_same_canonical_city(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    # Arjun Verma's raw resume says "Bombay"
    assert _by_name(backend, "Arjun Verma")["location"] == "Mumbai"


def test_bangalore_resolves_to_bengaluru(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    assert _by_name(backend, "Rohan Iyer")["location"] == "Bengaluru"


def test_hyphenated_city_matches_space_separated_gazetteer_form(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    assert _by_name(backend, "Kavya Reddy")["location"] == "Navi Mumbai"


def test_location_filter_unifies_alias_spellings(synthetic):
    # A recruiter searching "Mumbai" must find the candidate whose resume
    # literally said "Bombay" too -- that's the whole point of aliasing.
    backend = C.get_matched_candidates(JOB_BACKEND)
    spec = FilterSpec(logic="AND", filters=[Filter(field="location", operator="equals", value="Mumbai")])
    out = {c["name"] for c in apply_spec(backend, spec)}
    assert "Arjun Verma" in out  # was "Bombay"
    assert "Kavya Reddy" not in out  # "Navi Mumbai" is a different city


def test_missing_location_does_not_crash_and_excludes_from_location_filter(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    sameer = _by_name(backend, "Sameer Khan")
    assert sameer.get("location") is None or "location" not in sameer
    spec = FilterSpec(logic="AND", filters=[Filter(field="location", operator="equals", value="Mumbai")])
    out = {c["name"] for c in apply_spec(backend, spec)}
    assert "Sameer Khan" not in out


# --------------------------------------------------------------------------- #
# Employment gap: default-to-zero + boundary values
# --------------------------------------------------------------------------- #
def test_no_gap_defaults_to_zero_not_missing(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    assert _by_name(backend, "Meera Pillai")["employment_gap_months"] == 0


def test_gap_boundary_values_are_exact(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    assert _by_name(backend, "Sanjay Kulkarni")["employment_gap_months"] == 6
    assert _by_name(backend, "Farhan Sheikh")["employment_gap_months"] == 12


def test_gap_lte_boundary_is_inclusive_not_off_by_one(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    spec = FilterSpec(logic="AND", filters=[Filter(field="employment_gap_months", operator="lte", value=6)])
    out = {c["name"] for c in apply_spec(backend, spec)}
    assert "Sanjay Kulkarni" in out  # exactly 6, "lte 6" must include it
    assert "Farhan Sheikh" not in out  # 12 > 6


# --------------------------------------------------------------------------- #
# Education: multi-degree max-rank, including full-length "Doctor of
# Philosophy" phrasing (a real bug found and fixed during this dataset's
# construction -- was previously unrecognized as any degree level at all).
# --------------------------------------------------------------------------- #
def test_doctor_of_philosophy_phrasing_ranks_as_doctorate(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    assert _by_name(backend, "Karthik Subramaniam")["education"] == "Doctorate"


def test_highest_of_multiple_degrees_wins(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    # Rajesh Iyer has a Bachelor's AND a PhD -- must resolve to the PhD.
    assert _by_name(backend, "Rajesh Iyer")["education"] == "Doctorate"


# --------------------------------------------------------------------------- #
# Company tier: placeholder-company guard (a real false positive found and
# fixed during construction: real junk rows in company_ranks.json for
# "self employed" and "startup" were being matched as if they were real,
# ranked employers).
# --------------------------------------------------------------------------- #
def test_freelancer_placeholder_company_gets_no_fake_tier(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    assert _by_name(backend, "Rahul Mehta").get("company_tier") is None


# --------------------------------------------------------------------------- #
# Skill matching: naming-variant canonicalization (a real gap found and
# fixed during construction) and the java/javascript-style exact-token
# substring trap, generalized to "C" vs "C++".
# --------------------------------------------------------------------------- #
def test_skill_naming_variants_all_canonicalize_to_one_spelling(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    fatima = _by_name(backend, "Fatima Khan")
    # react.js / ReactJS / React JS must collapse to one consistent entry.
    assert fatima["skills"].count("React") == 1
    spec = FilterSpec(logic="AND", filters=[Filter(field="skill", operator="contains", value="React")])
    out = {c["name"] for c in apply_spec(backend, spec)}
    assert "Fatima Khan" in out


def test_bare_c_does_not_match_cpp_via_substring(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    spec = FilterSpec(logic="AND", filters=[Filter(field="skill", operator="contains", value="C++")])
    out = {c["name"] for c in apply_spec(backend, spec)}
    # Aditya Rao has "C" (embedded C), not "C++" -- must not match.
    assert "Aditya Rao" not in out


# --------------------------------------------------------------------------- #
# Certification: substring match against a messy multi-cert blob (mirrors
# the real dataset's Splunk-style pattern).
# --------------------------------------------------------------------------- #
def test_certification_matches_inside_a_messy_multi_cert_blob(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    spec = FilterSpec(logic="AND", filters=[Filter(field="certification", operator="contains", value="Databricks")])
    out = {c["name"] for c in apply_spec(backend, spec)}
    assert "David Chen" in out  # his one cert entry is a blob naming 3 different certs


# --------------------------------------------------------------------------- #
# Skills stored under the OTHER real key ("skills" instead of
# "skillsNormalized") -- confirms the fallback still works.
# --------------------------------------------------------------------------- #
def test_skills_field_fallback_when_skillsnormalized_is_absent(synthetic):
    backend = C.get_matched_candidates(JOB_BACKEND)
    divya = _by_name(backend, "Divya Menon")
    assert "Swift" in divya["skills"]


# --------------------------------------------------------------------------- #
# LOOKUP ambiguity: two candidates with the EXACT same full name.
# --------------------------------------------------------------------------- #
def test_exact_full_name_collision_is_ambiguous_not_silently_resolved(synthetic):
    from app.core.lookup import resolve_candidate
    backend = C.get_matched_candidates(JOB_BACKEND)
    priyas = [c for c in backend if c["name"] == "Priya Sharma"]
    assert len(priyas) == 2  # both real, distinct people
    res = resolve_candidate("Priya Sharma", priyas)
    # Must NOT silently pick one -- an exact duplicate name is genuinely
    # ambiguous and has to be flagged as such, not guessed.
    assert res.candidate is None
    assert res.ambiguous_names is not None
