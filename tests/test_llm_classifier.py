"""Tests for the general classify-with-mandatory-uncertainty mechanism.
Uses a fake llm_call (no real Ollama needed) so these are fast and
deterministic -- the whole point of this module is to be usable this way at
query time (cache-only), with real LLM calls confined to an explicit warmup
step tested separately against the live model."""
from __future__ import annotations

import json
import os
import tempfile

from app.core.llm_classifier import PersistentCache, UNKNOWN, classify_new_values


def _fake_llm(classifications: dict[str, str]):
    """Returns a llm_call that answers according to a fixed mapping,
    regardless of the actual prompt -- for controlled unit testing."""
    def _call(prompt: str, schema: dict, timeout: float) -> str:
        return json.dumps({
            "classifications": [
                {"name": name, "category": cat} for name, cat in classifications.items()
            ]
        })
    return _call


def test_known_values_classified_correctly():
    out = classify_new_values(
        ["Google", "Infosys"],
        dimension_description="product vs service",
        categories=["Product", "Service"],
        llm_call=_fake_llm({"Google": "Product", "Infosys": "Service"}),
    )
    assert out == {"Google": "Product", "Infosys": "Service"}


def test_value_missing_from_llm_response_defaults_to_unknown():
    # The model dropped "Mystery Corp" from its response entirely -- must
    # not crash or silently omit it, must default to UNKNOWN.
    out = classify_new_values(
        ["Google", "Mystery Corp"],
        dimension_description="product vs service",
        categories=["Product", "Service"],
        llm_call=_fake_llm({"Google": "Product"}),
    )
    assert out == {"Google": "Product", "Mystery Corp": UNKNOWN}


def test_infrastructure_failure_leaves_values_unclassified_not_cached_as_unknown():
    # A malformed response / timeout / connection error is a DIFFERENT thing
    # from the model genuinely saying "I don't know" -- must not crash, but
    # also must NOT cache these as a real Unknown classification (that would
    # permanently hide a company the model could classify fine on a retry
    # once whatever infrastructure problem clears). Confirmed to matter in
    # practice: a batch timed out under real GPU contention from a
    # concurrent query.
    def _broken_call(prompt, schema, timeout):
        return "not valid json at all"
    out = classify_new_values(
        ["Google", "Infosys"],
        dimension_description="product vs service",
        categories=["Product", "Service"],
        llm_call=_broken_call,
        max_retries=1,
    )
    assert out == {}  # left unclassified -- next warmup run will retry them


def test_transient_failure_recovers_on_retry():
    import requests
    calls = {"n": 0}
    def _flaky_call(prompt, schema, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.Timeout("simulated timeout")
        return json.dumps({"classifications": [{"name": "Google", "category": "Product"}]})

    out = classify_new_values(
        ["Google"], dimension_description="x", categories=["Product", "Service"],
        llm_call=_flaky_call, max_retries=2,
    )
    assert out == {"Google": "Product"}
    assert calls["n"] == 2


def test_batching_splits_large_input_and_still_covers_everything():
    values = [f"Company {i}" for i in range(75)]
    calls = []

    def _call(prompt, schema, timeout):
        calls.append(prompt)
        names = [line for line in prompt.split("\n") if line.startswith("Company")]
        return json.dumps({
            "classifications": [{"name": n, "category": "Service"} for n in names]
        })

    out = classify_new_values(
        values, dimension_description="x", categories=["Product", "Service"],
        llm_call=_call, batch_size=30,
    )
    assert len(calls) == 3  # 75 values / batch_size 30 -> 3 batches
    assert len(out) == 75
    assert all(v == "Service" for v in out.values())


def test_empty_input_makes_no_llm_call():
    calls = []
    def _call(prompt, schema, timeout):
        calls.append(1)
        return "{}"
    out = classify_new_values([], dimension_description="x", categories=["A"], llm_call=_call)
    assert out == {}
    assert calls == []


def test_persistent_cache_survives_reload():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cache.json")
        cache = PersistentCache(path)
        cache.update({"google": "Product"})

        # Simulate a fresh process reading the same file.
        reloaded = PersistentCache(path)
        assert reloaded.get_all() == {"google": "Product"}


def test_persistent_cache_missing_file_starts_empty():
    cache = PersistentCache("/nonexistent/path/cache.json")
    assert cache.get_all() == {}
