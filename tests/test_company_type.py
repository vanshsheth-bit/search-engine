"""Tests for the company_type module -- the query-time (cache-only) path
specifically, since that's what runs on every real request. The warmup path
(actually calling the LLM) is exercised live in scripts/warm_company_types.py,
not here."""
from __future__ import annotations

import app.core.company_type as CT


def test_company_types_for_uses_only_the_cache(monkeypatch):
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {
        "google": "Product", "infosys": "Service",
    })
    assert CT.company_types_for(["Google", "Infosys"]) == ["Product", "Service"]


def test_candidate_with_no_cached_companies_gets_empty_list(monkeypatch):
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {})
    assert CT.company_types_for(["Some Company"]) == []


def test_unknown_classification_is_omitted_not_included(monkeypatch):
    # Same semantics as any other missing-data field: Unknown means "no
    # classification available", not a real value to match against.
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {"mystery corp": "Unknown"})
    assert CT.company_types_for(["Mystery Corp"]) == []


def test_multiple_companies_with_different_types_both_kept(monkeypatch):
    # A candidate who worked at both a service company and a product company
    # genuinely has both -- no "best" winner like company_tier has.
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {
        "infosys": "Service", "google": "Product",
    })
    assert CT.company_types_for(["Infosys", "Google"]) == ["Product", "Service"]


def test_empty_company_list_returns_empty():
    assert CT.company_types_for([]) == []


def test_warm_cache_skips_already_cached_names(monkeypatch):
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {"google": "Product"})
    calls = []

    def fake_classify(values, **kwargs):
        calls.append(values)
        return {v: "Service" for v in values}

    monkeypatch.setattr(CT, "classify_new_values", fake_classify)
    monkeypatch.setattr(CT._cache(), "update", lambda d: None)

    n = CT.warm_cache(["Google", "New Company"])
    assert n == 1
    assert calls == [["New Company"]]  # Google skipped -- already cached


def test_warm_cache_with_nothing_new_makes_no_llm_call(monkeypatch):
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {"google": "Product"})
    calls = []
    monkeypatch.setattr(CT, "classify_new_values", lambda *a, **k: calls.append(1) or {})
    n = CT.warm_cache(["Google"])
    assert n == 0
    assert calls == []


# --------------------------------------------------------------------------- #
# Industry-based fallback -- this is what makes classification scale to
# thousands of companies: most real companies have no direct classification
# of their own (correctly Unknown -- the model has no specific knowledge of
# a small/regional business), but DO have a real industry on file, and the
# ~130 distinct industries are a small, tractable set to classify once.
# --------------------------------------------------------------------------- #
def test_industry_fallback_used_when_no_direct_classification(monkeypatch):
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {})  # no direct answer for this company
    monkeypatch.setattr(CT._industry_cache(), "get_all", lambda: {"computer software": "Product"})
    result = CT.company_types_for(
        ["Some Obscure Startup"],
        industry_lookup={"some obscure startup": "computer software"},
    )
    assert result == ["Product"]


def test_direct_classification_wins_over_industry_inference(monkeypatch):
    # A specific, direct answer (e.g. a manual correction) is more
    # authoritative than a generic industry-level guess -- must win even
    # when both are available and disagree.
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {"acme corp": "Service"})
    monkeypatch.setattr(CT._industry_cache(), "get_all", lambda: {"computer software": "Product"})
    result = CT.company_types_for(
        ["Acme Corp"], industry_lookup={"acme corp": "computer software"},
    )
    assert result == ["Service"]


def test_industry_classified_unknown_still_omitted(monkeypatch):
    # An industry that genuinely isn't a product-vs-service concept (e.g.
    # banking) correctly classifies as Unknown -- must not surface as a
    # real value.
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {})
    monkeypatch.setattr(CT._industry_cache(), "get_all", lambda: {"banking": "Unknown"})
    result = CT.company_types_for(
        ["Some Bank"], industry_lookup={"some bank": "banking"},
    )
    assert result == []


def test_no_industry_lookup_provided_skips_industry_fallback_entirely(monkeypatch):
    # Callers that don't pass industry_lookup at all (e.g. tests, or a
    # context where it's genuinely unavailable) must not crash -- just no
    # industry-based fallback happens.
    monkeypatch.setattr(CT._cache(), "get_all", lambda: {})
    result = CT.company_types_for(["Some Company"])
    assert result == []


def test_warm_industry_cache_is_independent_of_company_cache(monkeypatch):
    monkeypatch.setattr(CT._industry_cache(), "get_all", lambda: {})
    calls = []

    def fake_classify(values, **kwargs):
        calls.append(values)
        return {v: "Product" for v in values}

    monkeypatch.setattr(CT, "classify_new_values", fake_classify)
    monkeypatch.setattr(CT._industry_cache(), "update", lambda d: None)

    n = CT.warm_industry_cache(["computer software", "banking"])
    assert n == 2
    assert calls == [["computer software", "banking"]]
