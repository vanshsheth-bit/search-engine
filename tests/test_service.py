"""Service-layer tests using a fake LLM (no Ollama needed)."""
from __future__ import annotations

from app.core.service import FilterService
from app.core.session import InMemorySessionStore
from app.models.schemas import LLMOutput, Filter

# A real jdId from the real matched-candidates dataset (124 candidates,
# includes real Mumbai/Python candidates) -- see app/core/candidates.py.
JOB = "6a8c26ee15f64740b81997da"


class FakeLLM:
    """Returns a scripted LLMOutput regardless of input."""
    def __init__(self, output: LLMOutput):
        self.output = output
        self.calls = 0

    def translate(self, query, current_filters):
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
    assert resp.showing == 25
    assert resp.total == 124
    assert len(resp.chips) == 1


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
    # First: has Python (9 of the 124 real candidates for this job)
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains",
                                    value="python")])),
        store=store,
    )
    r1 = svc1.filter_by_query("python", job_id=JOB, session_id="s1")
    assert r1.showing == 9

    # Then: add location Mumbai ; should AND with existing Python -> only 1
    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals",
                                    value="Mumbai")])),
        store=store,
    )
    r2 = svc2.filter_by_query("mumbai", job_id=JOB, session_id="s1")
    assert r2.showing == 1
    assert len(r2.filters) == 2


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
    assert r1.showing == 9
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
