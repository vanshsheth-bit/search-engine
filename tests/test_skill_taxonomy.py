"""Tests for skill-concept expansion against merged_tools.json. Requires the
real merged_tools.json at repo root (gitignored real data, same as the
candidate datasets) -- skipped if it's not present in this checkout."""
from __future__ import annotations

import os

import pytest

from app.core.skill_taxonomy import (
    canonicalize, expand_skill_filters, expand_skill_term, is_known_tool,
    related_terms_for,
)
from app.models.schemas import Filter

_ROOT = os.path.join(os.path.dirname(__file__), "..")
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(_ROOT, "merged_tools.json")),
    reason="merged_tools.json not present in this checkout",
)


def test_known_concept_expands_with_related_tools():
    result = expand_skill_term("machine learning")
    assert result is not None
    lowered = {r.lower() for r in result}
    assert "machine learning" in lowered
    assert "tensorflow" in lowered
    assert "pytorch" in lowered


def test_expand_skill_term_excludes_near_duplicate_aliases_and_generic_libs():
    # Real, reported live bug: a bare "machine learning" mention (v2's
    # match_mode="expand", see taxonomy._tool_to_filter -- calls this
    # function DIRECTLY, with none of expand_skill_filters's is_known_tool
    # guarding) expanded to 28 items and rendered as a bewildering "machine
    # learning +27 more" checklist row. 17 of those were near-duplicate
    # self-referential ALIASES ("ML pipeline", "ML system", "predictive
    # modeling", "supervised learning", ...) contributing zero matching
    # value -- every candidate's own skill list is already canonicalized at
    # load time (candidates._adapt_resume), so a resume saying "ML pipeline"
    # is already stored as "machine learning"; the alias never needs to be
    # in the filter's own value list. NumPy (a _GENERIC_SUPPORT_LIBS entry)
    # was also present despite being useless standalone signal, same reason
    # related_terms_for already excludes it.
    result = expand_skill_term("machine learning")
    lowered = {r.lower() for r in result}
    assert "ml pipeline" not in lowered
    assert "ml system" not in lowered
    assert "predictive modeling" not in lowered
    assert "supervised learning" not in lowered
    assert "numpy" not in lowered
    # Real signal must survive -- this isn't just blanket truncation.
    assert "machine learning" in lowered
    assert "tensorflow" in lowered
    assert "pytorch" in lowered
    assert len(result) < 15  # was 28 before the fix


def test_related_terms_for_never_treats_a_different_language_as_related():
    # Real, reported live bug: "give me guy with python experience in high
    # tier company" surfaced a mechanical/controls engineer whose ONLY point
    # of contact with "Python" was knowing MATLAB -- a completely different
    # language, not a Python library or framework -- flagged "matched via a
    # related tool". The taxonomy's weighted related_tools scores treat two
    # languages used in similar problem domains (Python/MATLAB both showing
    # up in numerical-analysis work) as "related" the same way it treats
    # genuine sibling libraries (PyTorch/TensorFlow) -- but a recruiter who
    # names a specific language means that language, not "something in the
    # same general category". See skill_taxonomy._PROGRAMMING_LANGUAGES.
    _exact, related = related_terms_for("Python")
    for other_language in ("c++", "c#", "matlab", "lua", "julia", "r"):
        assert other_language not in related, (
            f"{other_language!r} must not count as related to Python -- "
            "it's a different language, not a Python library/framework"
        )
    # Genuine Python ecosystem tools (frameworks built ON Python) are
    # unaffected -- only language-vs-language relations are excluded.
    assert "django" in related
    assert "flask" in related

    # Sanity: the exclusion is symmetric-by-construction (both sides must be
    # in _PROGRAMMING_LANGUAGES) but does not over-fire on a genuine
    # non-language related pair sharing a domain with a language, e.g.
    # PyTorch's real sibling libraries stay intact.
    _exact2, related2 = related_terms_for("PyTorch")
    assert "tensorflow" in related2


def test_wrapped_in_generic_noise_word_resolves_same_as_the_bare_term():
    # Real, reported live bug: "knows machine learning concepts" resolved
    # to a literal, unrecognized "machine learning concepts" skill filter
    # that matched ZERO candidates, even though "machine learning" alone
    # (the recruiter's actual meaning) correctly expands via the real
    # taxonomy relationship. A recruiter routinely wraps a real skill/
    # concept name this way ("Python skills", "AWS knowledge") -- the
    # taxonomy only ever indexes the bare term, so this must be recognized
    # at the resolution layer (canonicalize/is_known_tool/expand_skill_term
    # -- see _resolve_canonical), not by hoping the LLM always strips it.
    assert canonicalize("machine learning concepts") == "machine learning"
    assert canonicalize("Python skills") == "Python"
    assert is_known_tool("machine learning concepts") is True
    expanded = expand_skill_term("machine learning concepts")
    assert expanded is not None
    assert expanded[0] == "machine learning"
    assert "tensorflow" in {v.lower() for v in expanded}
    # Must NOT fire for a genuinely different real skill entry that simply
    # happens to end in one of the noise words -- "Communication" is not a
    # known tool, so stripping "Skills" must not manufacture a false match.
    assert canonicalize("Communication Skills") == "Communication Skills"
    assert is_known_tool("Communication Skills") is False


def test_typo_in_a_real_skill_name_still_resolves():
    # Real, reported live bug: "knows muchine learning" (one substituted
    # letter) resolved to a literal, unrecognized filter that matched ZERO
    # candidates, the same failure mode as the noise-word-wrapper bug above
    # but for a genuine typo instead of a wrapper word. Conservative fix
    # (see _fuzzy_typo_match): a small bounded edit-distance match against
    # KNOWN CANONICAL NAMES ONLY, applied only when there's one single
    # unambiguous closest match AND the term is long enough (>= 10 normalized
    # characters) that coincidental proximity to an unrelated entry is
    # implausible.
    assert canonicalize("muchine learning") == "machine learning"
    assert canonicalize("Kuberentes") == "Kubernetes"  # exactly at the 10-char floor
    assert is_known_tool("muchine learning") is True
    # Must NOT fire below the length floor -- REAL regression this pins:
    # "safety" (6 chars, an ordinary complete English word) fuzzy-matched
    # an obscure, unrelated taxonomy entry ("SAFETI") at edit distance 1
    # purely by coincidence before the floor was raised from 5 to 10,
    # wrongly making is_known_tool("safety") true. A short word has too
    # little "surface area" for edit-distance proximity to mean anything
    # among ~15,000 canonical names -- short single-word typo tolerance
    # ("Pythom" -> "Python") is a real, accepted loss from this trade-off.
    assert canonicalize("safety") == "safety"
    assert is_known_tool("safety") is False
    assert canonicalize("Pythom") == "Pythom"
    # Must NOT fire for a real, different, unrelated PHRASE that just
    # happens to be close-ish in length either, even above the floor.
    assert canonicalize("Communication Skills") == "Communication Skills"
    assert canonicalize("xyzabc123notreal456") == "xyzabc123notreal456"


def test_fuzzy_typo_match_never_applies_to_a_noise_stripped_form():
    # Second real regression found while testing: fuzzy-matching the
    # STRIPPED form (not just the original term) let "Analytical Skills" --
    # a common resume soft-skill phrase -- strip to "Analytical", which
    # then fuzzy-matched "Analytica" (an obscure, unrelated BI tool) at
    # edit distance 1, purely by coincidence. Stacking noise-stripping and
    # typo-tolerance compounds the odds of a coincidental collision past
    # what either alone produces -- fixed by restricting _fuzzy_typo_match
    # to the ORIGINAL term only, never a stripped one (see
    # _resolve_canonical's docstring for the accepted trade-off this
    # implies: a typo INSIDE a noise-wrapped phrase is no longer caught).
    assert canonicalize("Analytical Skills") == "Analytical Skills"
    assert is_known_tool("Analytical Skills") is False


def test_unknown_term_returns_none_not_empty_list():
    # None (not []) signals "taxonomy has nothing to say", distinct from a
    # real hit that just happens to have no strong related tools.
    assert expand_skill_term("some-made-up-tool-xyz-123") is None


def test_single_tool_query_is_not_expanded_via_related_tools():
    # A plain "contains" filter for one specific, unambiguous tool must stay
    # precise -- must NOT pull in "related" (but different/competing) tools
    # like Django or Angular just because they share a taxonomy entry.
    f = Filter(field="skill", operator="contains", value="Python")
    out = expand_skill_filters([f])[0]
    assert out.operator == "contains"
    assert out.value == "Python"  # unchanged, not expanded into a list


def test_umbrella_concept_query_passes_through_when_not_a_recognized_tool():
    # "devops" is a genuine umbrella concept the taxonomy has NO tool entry
    # for at all (confirmed: expand_skill_term("devops") is None) -- exactly
    # like "cloud"/"frontend". The model's own proposed terms (per
    # prompt.py's rule 3 few-shot) pass through canonicalized, unchanged in
    # count -- there's nothing in the taxonomy to augment them with.
    f = Filter(field="skill", operator="in",
               value=["devops", "Kubernetes", "Docker", "Terraform", "Jenkins", "Ansible", "CI/CD"])
    out = expand_skill_filters([f])[0]
    assert out.operator == "in"
    lowered = {v.lower() for v in out.value}
    assert lowered == {"devops", "kubernetes", "docker", "terraform", "jenkins", "ansible", "ci/cd"}


def test_same_field_or_of_named_tools_is_not_expanded():
    # REAL BUG, confirmed live: "Python or Java" -- both genuinely specific,
    # already-recognized tools the recruiter explicitly named (prompt.py's
    # rule 4, same-field OR -> one "in" filter) -- must NOT be expanded via
    # the concept-augmentation path just because it shares the "in" operator
    # shape with a genuine umbrella-concept filter. Before this fix,
    # expand_skill_term("Python") pulled in 74 near-duplicate ALIAS entries
    # ("py", "cpython", "python3", ...) alongside genuine related tools,
    # silently turning a precise 2-item requirement into a 75-item one that
    # also DROPPED the "or Java" distinction into the same blob.
    f = Filter(field="skill", operator="in", value=["Python", "Java"])
    out = expand_skill_filters([f])[0]
    assert out.value == ["Python", "Java"]


def test_umbrella_concept_that_coincidentally_names_a_real_tool_still_does_not_expand():
    # "machine learning" is, unusually, ALSO its own taxonomy tool entry
    # (confirmed by direct inspection of data/Engineering.json -- no field
    # in this dataset distinguishes a genuine broad concept from a specific
    # product; both are plain tools_by_id entries with identical shape).
    # Since it can't be told apart from a same-field OR where every term
    # happens to be a recognized tool (e.g. "Python or Java"), it's treated
    # the same way: canonicalized, not re-expanded via related_tools. This
    # is a deliberate trade-off, not an oversight -- see
    # expand_skill_filters' docstring for why avoiding the same-field-OR
    # bug (the far more common real pattern) wins over preserving
    # augmentation for this narrower, coincidental case.
    f = Filter(field="skill", operator="in",
               value=["machine learning", "TensorFlow", "PyTorch"])
    out = expand_skill_filters([f])[0]
    lowered = {v.lower() for v in out.value}
    assert lowered == {"machine learning", "tensorflow", "pytorch"}


def test_canonicalize_resolves_alias_to_canonical_spelling():
    canon = canonicalize("pytorch framework")  # a known alias, if present
    # Either it resolves to the canonical "PyTorch" spelling, or (if that
    # exact alias isn't in the taxonomy) it's returned unchanged -- either
    # way canonicalize must never raise or return something falsy.
    assert canon


def test_non_skill_filters_are_untouched():
    f = Filter(field="location", operator="equals", value="Mumbai")
    out = expand_skill_filters([f])[0]
    assert out == f


def test_fuzzy_false_refuses_to_typo_correct_stored_resume_skills():
    # The bulk resume-ingestion path passes fuzzy=False (see
    # candidates._adapt_resume). Measured across all 1,807 distinct real
    # skill strings in this dataset, the typo tier changed only 9 of them
    # -- and 8 of those 9 were CORRUPTIONS of a perfectly valid skill into
    # an unrelated one. These are real cases from that audit: a stored
    # resume skill must survive ingestion unchanged rather than being
    # silently "corrected" into a different product/concept.
    for stored in (
        "google sites",        # -> "Google Slides": a different Google product
        "sql injection",       # -> "jSQL Injection": a vuln class, not that tool
        "lean six sigma",      # -> "GoLeanSixSigma": a methodology, not a vendor
        "code deployment",     # -> "model deployment": a different thing entirely
        "orchestration",       # -> "Orchestrator": a concept, not that product
        "benchmarks",          # -> "Benchmark.js": an ordinary word, not that lib
    ):
        assert canonicalize(stored, fuzzy=False) == stored

    # The query side (default fuzzy=True) is deliberately unchanged -- that
    # is where a recruiter's genuine typo needs catching, and where a wrong
    # guess is visible and self-correcting rather than baked into the data.
    assert canonicalize("muchine learning").lower() == "machine learning"


def test_is_known_tool_honours_the_same_no_fuzzy_opt_out():
    # Same reasoning as above, via the other entry point onto
    # _resolve_canonical -- candidates._skill_years_from_experience gates on
    # this, so a fuzzily-"recognized" non-tool would fabricate years for it.
    assert is_known_tool("benchmarks", fuzzy=False) is False
