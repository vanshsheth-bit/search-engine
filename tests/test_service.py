"""Service-layer tests using a fake LLM (no Ollama needed)."""
from __future__ import annotations

import app.core.service as service_module
from app.core.service import FilterService
from app.core.session import InMemorySessionStore
from app.models.schemas import LLMOutput, Filter

# A real jdId from the real matched-candidates dataset (111 candidates
# after dedup-by-real-person -- see _identity_key in candidates.py --
# includes real Mumbai/Python candidates). See app/core/candidates.py.
JOB = "6a8c26ee15f64740b81997da"


class FakeLLM:
    """Returns a scripted LLMOutput regardless of input."""
    def __init__(self, output: LLMOutput):
        self.output = output
        self.calls = 0

    def translate(self, query, current_filters, history=None):
        self.calls += 1
        return self.output


def make_service(output: LLMOutput) -> FilterService:
    return FilterService(llm=FakeLLM(output), store=InMemorySessionStore())


def test_ok_flow():
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals",
                                    value="Mumbai")])
    svc = make_service(out)
    resp = svc.filter_by_query("mumbai", job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    assert resp.showing == 22
    assert resp.total == 111
    assert len(resp.chips) == 1


def test_domain_filter_finds_real_candidate_via_classified_experience():
    # Real, verified on this dataset: Manohar Patil's classified subdomains
    # include "Software Engineering" (from experience_index/classifications.jsonl,
    # a real classifier run over his actual experience text, not a guess).
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="domain", operator="contains", value="software engineering")])
    svc = make_service(out)
    resp = svc.filter_by_query("candidates with software engineering background",
                                job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    names = {c["name"] for c in resp.candidates}
    assert "Manohar Patil" in names


def test_colloquial_country_name_resolves_to_real_matches():
    # The LLM is allowed to emit a colloquial short form ("USA") -- service.py
    # resolves it to the exact "United States" spelling candidates are
    # tagged with (see _canonicalize_country_filter) before matching, same
    # as if it had emitted the full name itself.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="country", operator="equals", value="USA")])
    svc = make_service(out)
    resp = svc.filter_by_query("candidates in the usa", job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    assert resp.showing == 2
    assert resp.filters[0].value == "United States"


def test_clarify_flow():
    out = LLMOutput(intent="CLARIFY", question="How many years?",
                    options=["2+ years", "3+ years"])
    svc = make_service(out)
    resp = svc.filter_by_query("experienced", job_id="123", session_id="s1")
    assert resp.status == "clarify"
    assert resp.options == ["2+ years", "3+ years"]


def test_unsupported_flow():
    out = LLMOutput(intent="UNSUPPORTED_FILTER",
                    message="Salary data not available.")
    svc = make_service(out)
    resp = svc.filter_by_query("high salary", job_id="123", session_id="s1")
    assert resp.status == "unsupported"
    assert "Salary" in resp.message


def test_no_match_flow():
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill_experience", operator="gte",
                                    skill="Python", value=99)])
    svc = make_service(out)
    resp = svc.filter_by_query("python 99 years", job_id=JOB, session_id="s1")
    assert resp.status == "no_match"
    assert resp.showing == 0
    assert resp.suggestions


def test_session_state_persists_and_merges():
    store = InMemorySessionStore()
    # First: has Python (11 of the 111 real candidates for this job, incl.
    # fuzzy taxonomy-related matches -- see FilterService._fuzzy_skill_matches)
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains",
                                    value="python")])),
        store=store,
    )
    r1 = svc1.filter_by_query("python", job_id=JOB, session_id="s1")
    assert r1.showing == 11

    # Then: add location Mumbai. This introduces a genuinely new field on
    # top of an already-active search, so it's a PendingCombine confirmation
    # rather than a silent merge (see PendingCombine's docstring) -- ask
    # first.
    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals",
                                    value="Mumbai")])),
        store=store,
    )
    r2 = svc2.filter_by_query("mumbai", job_id=JOB, session_id="s1")
    assert r2.status == "clarify"
    assert r2.options == ["Yes", "No"]

    # Confirming resolves deterministically (no LLM call) to the pre-merged
    # spec -> AND with existing Python -> only 1.
    svc3 = FilterService(llm=svc2.llm, store=store)
    r3 = svc3.filter_by_query("yes", job_id=JOB, session_id="s1")
    assert r3.showing == 1
    assert len(r3.filters) == 2


def test_lookup_answers_from_real_data_after_narrowing_to_one():
    store = InMemorySessionStore()
    # Narrow to exactly one real candidate (Mumbai + python, confirmed
    # unique in this job's pool).
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals", value="Mumbai"),
                             Filter(field="skill", operator="contains", value="python")])),
        store=store,
    )
    r1 = svc1.filter_by_query("mumbai python", job_id=JOB, session_id="s1")
    assert r1.showing == 1

    # Follow-up: a question about that one candidate, not a new filter.
    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="LOOKUP", lookup_field="education")),
        store=store,
    )
    r2 = svc2.filter_by_query("what's his education level?", job_id=JOB, session_id="s1")
    assert r2.status == "answer"
    assert r1.candidates[0]["name"] in r2.message


def test_ambiguous_lookup_then_bare_name_reply_completes_it():
    store = InMemorySessionStore()
    # Narrow to several candidates (all with python), so a LOOKUP that
    # doesn't clearly name one of them is ambiguous.
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="python")])),
        store=store,
    )
    r1 = svc1.filter_by_query("python", job_id=JOB, session_id="s1")
    assert r1.showing == 11
    names = [c["name"] for c in r1.candidates]

    # Ambiguous lookup -- no candidate_ref given, several candidates shown.
    lookup_llm = FakeLLM(LLMOutput(intent="LOOKUP", lookup_field="education"))
    svc2 = FilterService(llm=lookup_llm, store=store)
    r2 = svc2.filter_by_query("what's their education?", job_id=JOB, session_id="s1")
    assert r2.status == "clarify"
    assert set(r2.options) == set(names)

    # Reply with just a name (as clicking an option would submit). This
    # must NOT go through the LLM at all -- it should resolve directly
    # against the pending lookup.
    svc3 = FilterService(llm=lookup_llm, store=store)
    r3 = svc3.filter_by_query(names[0], job_id=JOB, session_id="s1")
    assert r3.status == "answer"
    assert names[0] in r3.message
    assert lookup_llm.calls == 1  # only the first (ambiguous) call hit the LLM


def test_lookup_with_no_candidates_shown_is_honest():
    svc = make_service(LLMOutput(intent="LOOKUP", lookup_field="university"))
    resp = svc.filter_by_query("which college did he go to?", job_id=JOB, session_id="fresh")
    assert resp.status == "unsupported"
    assert "search for someone first" in resp.message.lower()


def test_clarify_reply_resolves_deterministically_without_llm():
    # Regression: clicking a CLARIFY option ("2+ years") used to be re-sent
    # to the LLM with zero memory of the question -- a bare fragment like
    # that isn't reliably interpretable in isolation, so it just asked the
    # same question again instead of ever producing a filter.
    store = InMemorySessionStore()
    clarify_llm = FakeLLM(LLMOutput(
        intent="CLARIFY", question="What minimum years of experience should I use?",
        options=["2+ years", "3+ years", "5+ years"],
        clarify_field="experience", clarify_operator="gte",
    ))
    svc1 = FilterService(llm=clarify_llm, store=store)
    r1 = svc1.filter_by_query("candidate with experience", job_id=JOB, session_id="s1")
    assert r1.status == "clarify"

    # Reply with an option (as clicking one would submit). A DIFFERENT fake
    # LLM that would return the wrong thing if it were ever called -- this
    # must resolve directly against the pending clarification instead.
    wrong_llm = FakeLLM(LLMOutput(intent="CLARIFY", question="asked again wrongly"))
    svc2 = FilterService(llm=wrong_llm, store=store)
    r2 = svc2.filter_by_query("2+ years", job_id=JOB, session_id="s1")
    assert r2.status in ("ok", "no_match")
    assert any(f.field == "experience" and f.operator == "gte" and f.value == 2 for f in r2.filters)
    assert wrong_llm.calls == 0  # never invoked -- resolved deterministically


def test_clarify_reply_falls_through_to_llm_when_not_a_plausible_answer():
    store = InMemorySessionStore()
    clarify_llm = FakeLLM(LLMOutput(
        intent="CLARIFY", question="What minimum years of experience should I use?",
        options=["2+ years", "3+ years", "5+ years"],
        clarify_field="experience", clarify_operator="gte",
    ))
    svc1 = FilterService(llm=clarify_llm, store=store)
    svc1.filter_by_query("candidate with experience", job_id=JOB, session_id="s2")

    # A reply with no number in it isn't a plausible answer -- must fall
    # through to a fresh LLM call rather than get stuck.
    fresh_llm = FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                filters=[Filter(field="location", operator="equals", value="Mumbai")]))
    svc2 = FilterService(llm=fresh_llm, store=store)
    r2 = svc2.filter_by_query("actually show me Mumbai instead", job_id=JOB, session_id="s2")
    assert fresh_llm.calls == 1
    assert any(f.field == "location" for f in r2.filters)


def test_reset_clears_state():
    store = InMemorySessionStore()
    svc = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals",
                                    value="Mumbai")])),
        store=store,
    )
    svc.filter_by_query("mumbai", job_id=JOB, session_id="s1")
    r = svc.filter_by_query("mumbai", job_id=JOB, session_id="s1", reset=True)
    # after reset only the new single filter is present
    assert len(r.filters) == 1


# --------------------------------------------------------------------------- #
# EXPERIENCE_SEARCH -- real candidate ids from JOB's matched pool, used to
# verify the job/filter-scoping logic (a hit for a candidate NOT in this
# job's real matched pool must never leak into the result).
# --------------------------------------------------------------------------- #
_REAL_ID_1 = "proc_a9875f1e-6420-4135-b830-f268e0d072a4"  # Manohar Patil, real, in JOB
_REAL_ID_2 = "proc_1b345db4-d5e3-4551-88ce-97b0a1cd297b"  # Ganesh B Shelke, real, in JOB
_FAKE_ID = "proc_00000000-0000-0000-0000-000000000000"    # not in any job's pool


def test_experience_search_reports_unavailable_when_index_not_built(monkeypatch):
    class _FakePaths:
        def exists(self):
            return False
    monkeypatch.setattr(service_module.experience_index, "IndexPaths", _FakePaths)

    out = LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="led a team")
    svc = make_service(out)
    resp = svc.filter_by_query("who led a team", job_id=JOB, session_id="s1")
    assert resp.status == "unsupported"
    assert "available" in resp.message.lower()


def test_experience_search_missing_query_asks_to_rephrase():
    out = LLMOutput(intent="EXPERIENCE_SEARCH")  # experience_query left unset
    svc = make_service(out)
    resp = svc.filter_by_query("something vague", job_id=JOB, session_id="s1")
    assert resp.status == "unsupported"
    assert "rephrase" in resp.message.lower() or "describe" in resp.message.lower()


def test_experience_search_scopes_to_real_job_matched_pool(monkeypatch):
    class _FakePaths:
        def exists(self):
            return True
    monkeypatch.setattr(service_module.experience_index, "IndexPaths", _FakePaths)

    def fake_search(query, top_k=200):
        return [
            {"candidate_id": _REAL_ID_1, "score": 0.81},
            {"candidate_id": _FAKE_ID, "score": 0.95},  # not in JOB -- must be dropped
        ]
    monkeypatch.setattr(service_module.experience_index, "search", fake_search)

    out = LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="led a team of engineers")
    svc = make_service(out)
    resp = svc.filter_by_query("who led a team of engineers", job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    ids = {c["id"] for c in resp.candidates}
    assert ids == {_REAL_ID_1}
    assert resp.candidates[0]["experience_match_score"] == 0.81


def test_experience_search_keeps_best_score_per_candidate(monkeypatch):
    class _FakePaths:
        def exists(self):
            return True
    monkeypatch.setattr(service_module.experience_index, "IndexPaths", _FakePaths)

    def fake_search(query, top_k=200):
        # Same candidate, two matching experience chunks -- best score wins.
        return [
            {"candidate_id": _REAL_ID_1, "score": 0.60},
            {"candidate_id": _REAL_ID_1, "score": 0.88},
        ]
    monkeypatch.setattr(service_module.experience_index, "search", fake_search)

    out = LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="built a payment system")
    svc = make_service(out)
    resp = svc.filter_by_query("who built a payment system", job_id=JOB, session_id="s1")
    assert resp.showing == 1
    assert resp.candidates[0]["experience_match_score"] == 0.88


def test_experience_search_no_hits_returns_no_match(monkeypatch):
    class _FakePaths:
        def exists(self):
            return True
    monkeypatch.setattr(service_module.experience_index, "IndexPaths", _FakePaths)
    monkeypatch.setattr(service_module.experience_index, "search", lambda query, top_k=200: [])

    out = LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="flew to the moon")
    svc = make_service(out)
    resp = svc.filter_by_query("who flew to the moon", job_id=JOB, session_id="s1")
    assert resp.status == "no_match"
    assert resp.showing == 0


def test_experience_search_drops_matches_below_similarity_floor(monkeypatch):
    # Confirmed live across two independent real queries: genuine matches
    # cluster ~0.62-0.67, but clearly unrelated text is already interleaved
    # in by ~0.61-0.62 (see FilterService._EXPERIENCE_MIN_SIMILARITY's
    # docstring for the full empirical basis) -- below that floor a
    # "match" must be dropped, not just ranked low.
    class _FakePaths:
        def exists(self):
            return True
    monkeypatch.setattr(service_module.experience_index, "IndexPaths", _FakePaths)

    floor = service_module.FilterService._EXPERIENCE_MIN_SIMILARITY
    below_floor = floor - 0.01
    above_floor = floor + 0.10

    def fake_search(query, top_k=200):
        return [
            {"candidate_id": _REAL_ID_1, "score": above_floor},
            {"candidate_id": _REAL_ID_2, "score": below_floor},
        ]
    monkeypatch.setattr(service_module.experience_index, "search", fake_search)

    out = LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="led a team")
    svc = make_service(out)
    resp = svc.filter_by_query("who led a team", job_id=JOB, session_id="s1")
    ids = {c["id"] for c in resp.candidates}
    assert ids == {_REAL_ID_1}


def test_experience_search_intersects_with_active_structured_filter(monkeypatch):
    class _FakePaths:
        def exists(self):
            return True
    monkeypatch.setattr(service_module.experience_index, "IndexPaths", _FakePaths)

    def fake_search(query, top_k=200):
        # Both real candidates match semantically...
        return [
            {"candidate_id": _REAL_ID_1, "score": 0.90},
            {"candidate_id": _REAL_ID_2, "score": 0.85},
        ]
    monkeypatch.setattr(service_module.experience_index, "search", fake_search)

    store = InMemorySessionStore()
    # First: a real structured filter narrows the active pool to just
    # Manohar Patil's location (confirmed via the real dataset -- see
    # test_lookup_answers_from_real_data_after_narrowing_to_one below for
    # the same "narrow to one real person" pattern).
    from app.core.candidates import get_matched_candidates
    manohar = next(c for c in get_matched_candidates(JOB) if c["id"] == _REAL_ID_1)
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals",
                                    value=manohar["location"])])),
        store=store,
    )
    r1 = svc1.filter_by_query(f"candidates in {manohar['location']}", job_id=JOB, session_id="s1")
    assert any(c["id"] == _REAL_ID_1 for c in r1.candidates)
    assert not any(c["id"] == _REAL_ID_2 for c in r1.candidates)

    # Then: EXPERIENCE_SEARCH on top -- only intersects with what's already
    # active, so Ganesh (who matched semantically but not the location) is
    # excluded even though experience_index.search found him too.
    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="led a team")),
        store=store,
    )
    r2 = svc2.filter_by_query("who led a team", job_id=JOB, session_id="s1")
    ids = {c["id"] for c in r2.candidates}
    assert _REAL_ID_1 in ids
    assert _REAL_ID_2 not in ids
