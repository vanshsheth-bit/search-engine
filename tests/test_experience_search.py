"""Tests for the experience_domain / experience_search integration -- the
wiring that connects the offline experience index (app/core/experience_index.py)
to the live chat pipeline (app/core/service.py, app/core/engine.py).

Two things this specifically guards against, both confirmed as real bugs
during development, not hypothetical:

1. `experience_domain` is stored as a LIST on each candidate (someone can
   have worked across several domains) -- "equals" compares against the
   whole list at once (str(list) == str(value)) and silently matches
   NOBODY even when the value is genuinely present. Confirmed against real
   data: 0 matches via "equals", 63 via "in" with the identical single
   value. Only "in"/"not_in" are safe, same convention company_type
   already established. See test_equals_silently_matches_nobody_on_a_list_field
   below -- it exists so nobody "fixes" the prompt back to "equals" without
   understanding why that's wrong.

2. `search_candidates()` must restrict to the candidate pool BEFORE
   ranking, not rank globally then filter -- otherwise a small job's real
   best match can get truncated away by irrelevant top_k noise from
   candidates outside that job entirely. See
   test_search_candidates_restricts_before_ranking.

No live Ollama or real candidate data required -- the embedding call is
monkeypatched to a fixed, deterministic vector space.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import experience_index
from app.core.engine import apply_spec
from app.models.schemas import Filter, FilterSpec


# ---------------------------------------------------------------------------
# Bug 1: equals vs in on a list-valued field
# ---------------------------------------------------------------------------

def test_equals_silently_matches_nobody_on_a_list_field():
    candidates = [
        {"id": "c1", "experience_domain": ["Sales", "Engineering"]},
        {"id": "c2", "experience_domain": ["Research / Science"]},
    ]
    equals_spec = FilterSpec(logic="AND", filters=[
        Filter(field="experience_domain", operator="equals", value="Sales"),
    ])
    # This is the trap, documented rather than silently "fixed" here --
    # equals on a list field compares str(list) to str(value) and matches
    # nothing. If engine.py's generic dispatch is ever changed to special-
    # case "equals" for list fields, this assertion should flip to True and
    # the prompt's "always use in, never equals" rule can be relaxed.
    assert apply_spec(candidates, equals_spec) == []


def test_in_is_the_correct_operator_for_a_single_domain_value():
    candidates = [
        {"id": "c1", "experience_domain": ["Sales", "Engineering"]},
        {"id": "c2", "experience_domain": ["Research / Science"]},
    ]
    in_spec = FilterSpec(logic="AND", filters=[
        Filter(field="experience_domain", operator="in", value=["Sales"]),
    ])
    result = apply_spec(candidates, in_spec)
    assert [c["id"] for c in result] == ["c1"]


def test_in_with_multiple_domains_is_an_or():
    candidates = [
        {"id": "c1", "experience_domain": ["Sales"]},
        {"id": "c2", "experience_domain": ["Research / Science"]},
        {"id": "c3", "experience_domain": ["HR/People"]},
    ]
    spec = FilterSpec(logic="AND", filters=[
        Filter(field="experience_domain", operator="in",
               value=["Sales", "Research / Science"]),
    ])
    result = apply_spec(candidates, spec)
    assert {c["id"] for c in result} == {"c1", "c2"}


def test_domain_matching_is_case_insensitive_but_not_substring():
    # Real data has both "PRESALES/SOLUTIONS" and "Presales/Solutions" as
    # distinct raw strings from the taxonomy -- matching must treat them as
    # the same domain (case-insensitive exact-token match), the same way
    # "skill" already does, so the taxonomy's casing inconsistency doesn't
    # silently fragment one domain into two unmatchable filter values.
    candidates = [{"id": "c1", "experience_domain": ["PRESALES/SOLUTIONS"]}]
    spec = FilterSpec(logic="AND", filters=[
        Filter(field="experience_domain", operator="in", value=["presales/solutions"]),
    ])
    assert [c["id"] for c in apply_spec(candidates, spec)] == ["c1"]

    # But "Sales" must NOT match "Presales/Solutions" as a substring --
    # atomic-token matching, same guarantee "skill" gives against
    # "java"/"javascript".
    candidates2 = [{"id": "c1", "experience_domain": ["Presales/Solutions"]}]
    spec2 = FilterSpec(logic="AND", filters=[
        Filter(field="experience_domain", operator="in", value=["Sales"]),
    ])
    assert apply_spec(candidates2, spec2) == []


# ---------------------------------------------------------------------------
# Bug 2: restrict-before-rank in search_candidates
# ---------------------------------------------------------------------------

def _write_fake_index(tmp_path, chunk_texts_by_candidate: dict[str, str], dim: int = 4):
    """Writes a tiny fake experience index: one chunk per candidate, whose
    embedding is READ DIRECTLY off a `#vec:a,b,c,d` suffix in the text
    (fully deterministic, no real embedding model involved)."""
    chunk_rows, vectors = [], []
    for i, (cid, text) in enumerate(chunk_texts_by_candidate.items()):
        marker = text.split("#vec:")[1]
        vec = np.array([float(x) for x in marker.split(",")], dtype=np.float32)
        vectors.append(vec)
        chunk_rows.append({
            "chunk_id": f"{cid}#exp0#chunk0", "experience_id": f"{cid}#exp0",
            "candidate_id": cid, "experience_index": 0, "chunk_index": 0,
            "n_chunks": 1, "n_chars": len(text), "text": text,
        })
    experience_index.write_embeddings(chunk_rows, np.stack(vectors), tmp_path)
    experience_index.write_classifications([], tmp_path)
    experience_index.write_manifest({"embedding_dim": dim}, tmp_path)


def test_search_candidates_restricts_before_ranking(tmp_path, monkeypatch):
    # Three candidates. c_outside is the closest match to the query overall,
    # but is NOT in this job's candidate pool. c_in_pool is a weaker but
    # real match that IS in the pool. A rank-then-restrict implementation
    # would let c_outside's dominance push nothing wrong here directly (it's
    # just excluded either way) -- the real risk this guards is a top_k CUT
    # happening before the pool restriction, which this test forces by
    # padding with many outside-pool candidates ranked above the real hit.
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    chunks = {"outside_1": "irrelevant text #vec:1,0,0,0"}
    for i in range(2, 30):  # 28 more outside-pool candidates, all closer than the real hit
        chunks[f"outside_{i}"] = f"irrelevant text {i} #vec:0.99,0.01,0,0"
    chunks["in_pool_1"] = "the real match #vec:0.5,0.5,0,0"

    _write_fake_index(tmp_path, chunks)

    monkeypatch.setattr(
        experience_index, "default_cache",
        lambda: type("FakeCache", (), {"embed": staticmethod(lambda texts, **kw: query_vec.reshape(1, -1))})(),
    )

    # top_k=5 -- if restriction happened AFTER ranking, the 28 closer
    # "outside" chunks would fill all 5 slots and in_pool_1 would never
    # appear, even though it's the only candidate actually in the pool.
    results = experience_index.search_candidates(
        "query", candidate_ids={"in_pool_1"}, top_k=5, directory=tmp_path,
    )
    assert [r["candidate_id"] for r in results] == ["in_pool_1"]


def test_search_candidates_returns_one_result_per_candidate(tmp_path, monkeypatch):
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(
        experience_index, "default_cache",
        lambda: type("FakeCache", (), {"embed": staticmethod(lambda texts, **kw: query_vec.reshape(1, -1))})(),
    )
    # Two chunks for the SAME candidate (multi-experience), different
    # scores -- only the best one should be returned, once.
    chunk_rows = [
        {"chunk_id": "c1#exp0#chunk0", "experience_id": "c1#exp0", "candidate_id": "c1",
         "experience_index": 0, "chunk_index": 0, "n_chunks": 1, "n_chars": 5, "text": "weak"},
        {"chunk_id": "c1#exp1#chunk0", "experience_id": "c1#exp1", "candidate_id": "c1",
         "experience_index": 1, "chunk_index": 0, "n_chunks": 1, "n_chars": 6, "text": "strong"},
    ]
    vectors = np.array([[0.1, 0.9, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
    experience_index.write_embeddings(chunk_rows, vectors, tmp_path)
    experience_index.write_classifications([], tmp_path)
    experience_index.write_manifest({"embedding_dim": 4}, tmp_path)

    results = experience_index.search_candidates(
        "query", candidate_ids={"c1"}, top_k=5, directory=tmp_path,
    )
    assert len(results) == 1
    assert results[0]["text"] == "strong"


def test_search_candidates_empty_pool_returns_nothing(tmp_path):
    assert experience_index.search_candidates("query", set(), top_k=5, directory=tmp_path) == []
