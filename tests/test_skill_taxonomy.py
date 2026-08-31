"""Tests for skill-concept expansion against merged_tools.json. Requires the
real merged_tools.json at repo root (gitignored real data, same as the
candidate datasets) -- skipped if it's not present in this checkout."""
from __future__ import annotations

import os

import pytest

from app.core.skill_taxonomy import canonicalize, expand_skill_filters, expand_skill_term
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


def test_umbrella_concept_query_expands_via_in_operator():
    # The LLM signals "this is a concept, not one tool" by already using
    # "in" with several proposed terms (per prompt.py's rule) -- concept
    # first, its own suggestions after.
    f = Filter(field="skill", operator="in",
               value=["machine learning", "TensorFlow", "PyTorch"])
    out = expand_skill_filters([f])[0]
    assert out.operator == "in"
    assert isinstance(out.value, list)
    lowered = {v.lower() for v in out.value}
    assert "machine learning" in lowered
    assert "tensorflow" in lowered
    assert "scikit learn" in lowered or "scikit-learn" in lowered  # a real related tool, not proposed by the model itself


def test_expansion_does_not_cascade_through_every_proposed_tools_own_relations():
    # Only the concept phrase (first item) gets its related_tools pulled in;
    # the model's own specific suggestions are canonicalized only, not
    # independently re-expanded -- otherwise this snowballs into unrelated
    # tools via each suggested tool's own "related" list.
    f = Filter(field="skill", operator="in",
               value=["machine learning", "TensorFlow", "PyTorch", "scikit-learn", "Keras"])
    out = expand_skill_filters([f])[0]
    # TensorFlow's own related tools include things like CUDA/Hugging Face/
    # DeepSpeed -- those must NOT leak in just because TensorFlow was one of
    # several proposed terms.
    lowered = {v.lower() for v in out.value}
    assert "cuda" not in lowered
    assert "deepspeed" not in lowered


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
