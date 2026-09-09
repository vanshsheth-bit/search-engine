"""Service-layer tests using a fake LLM (no Ollama needed)."""
from __future__ import annotations

import app.core.service as service_module
from app.core.service import FilterService
from app.core.session import InMemorySessionStore
from app.models.schemas import AlternativeGroup, LLMOutput, Filter

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


def test_company_size_query_strips_fabricated_company_type():
    """Deterministic safety net for a confirmed, repeatable model failure
    (see prompt.py rule 7a): a query naming company SIZE reliably got a
    fabricated "company_type" filter tacked on alongside the correct
    "company size isn't tracked" message -- confirmed live, byte-identical,
    across 6 separate attempts even after two rounds of prompt strengthening.
    Enforced in service.py instead, since prompt wording alone did not
    reliably stop it for this model. This test scripts the EXACT confirmed
    failure shape (FakeLLM standing in for the real model's actual bad
    output) and asserts the service layer corrects it -- not that the
    model gets it right, which it doesn't yet."""
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[
            Filter(field="experience", operator="gte", value=8),
            Filter(field="skill", operator="contains", value="Kubernetes"),
            Filter(field="company_type", operator="in", value=["Product", "Both"]),
        ],
        message="Company size isn't tracked, so that part couldn't be applied.",
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "8+ years, Kubernetes, at a large company.", job_id=JOB, session_id="s1",
    )
    fields = {f.field for f in resp.filters}
    assert "company_type" not in fields
    assert "company_tier" not in fields
    assert resp.message and "company size" in resp.message.lower()


def test_company_size_guard_does_not_touch_unrelated_queries():
    """The guard is keyed on the query text itself, not just on whether a
    company_type/tier filter is present -- a genuine, real company_type
    request must survive untouched even though it shares a field with the
    fabrication this guard exists to catch."""
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[Filter(field="company_type", operator="in", value=["Product", "Both"])],
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "candidates with product company experience", job_id=JOB, session_id="s1",
    )
    assert any(f.field == "company_type" for f in resp.filters)


def test_stale_filter_echo_triggers_clarify_instead_of_silent_repeat():
    """Deterministic safety net for a confirmed, repeatable model failure
    (3 separate live reproductions, see prompt.py rules 1c/1d): given a
    genuinely new query sharing no content with the currently active
    filters, the model sometimes just hands the SAME filters back
    unchanged instead of translating the new query -- confirmed live even
    with two rounds of prompt rules already in place. Scripts the exact
    confirmed failure shape (FakeLLM standing in for the real model's
    actual bad output) and asserts the service layer catches it."""
    store = InMemorySessionStore()
    stale_filters = [
        Filter(field="education", operator="gte", value="Bachelor"),
        Filter(field="skill", operator="contains", value="React"),
        Filter(field="skill", operator="contains", value="JavaScript"),
    ]

    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND", filters=stale_filters)),
        store=store,
    )
    r1 = svc1.filter_by_query(
        "Show me fresh graduates who know React and JavaScript", job_id=JOB, session_id="s1",
    )
    assert r1.status == "ok"

    # A genuinely unrelated new query, but the (fake) model hands back the
    # SAME filters unchanged -- the exact confirmed live failure shape.
    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    replace_all=True, filters=stale_filters)),
        store=store,
    )
    r2 = svc2.filter_by_query(
        "want a guy who has worked in fintech company for atleast 3years",
        job_id=JOB, session_id="s1",
    )
    assert r2.status == "clarify"
    # The stale filters must NOT have been silently applied as if correct
    # (r2.filters should still be the SAME state r1 already had -- the
    # clarify response preserves current filters, it doesn't discard them).
    key = lambda fs: {(f.field, str(f.value)) for f in fs}  # noqa: E731
    assert key(r2.filters) == key(r1.filters)


def test_genuine_repeat_is_not_flagged_as_stale_echo():
    """The guard must not false-fire on a genuine repeat -- if the new
    query actually names something the active filters are about, handing
    back the same filters is a perfectly legitimate, correct answer."""
    store = InMemorySessionStore()
    filters = [Filter(field="location", operator="equals", value="Mumbai")]

    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND", filters=filters)),
        store=store,
    )
    svc1.filter_by_query("candidates in Mumbai", job_id=JOB, session_id="s1")

    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND", filters=filters)),
        store=store,
    )
    r2 = svc2.filter_by_query(
        "still want Mumbai candidates please", job_id=JOB, session_id="s1",
    )
    assert r2.status == "ok"


def test_hard_filter_duplicate_of_preferred_is_stripped():
    """Confirmed live: given "prefer X", the model sometimes correctly adds
    X to preferred_filters AND separately leaves a hard copy of the exact
    same (field, value) in `filters` too -- the hard copy silently defeats
    the whole point of it being a preference. The service layer must strip
    that hard duplicate, keeping only the non-exclusionary preferred copy."""
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[
            Filter(field="skill", operator="contains", value="Python"),
            Filter(field="company_tier", operator="gte", value="High"),  # the fabricated hard duplicate
        ],
        preferred_filters=[Filter(field="company_tier", operator="gte", value="High")],
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "Python developers, preferably at a high-tier company",
        job_id=JOB, session_id="s1",
    )
    hard_fields = {f.field for f in resp.filters}
    assert "company_tier" not in hard_fields
    assert any(f.field == "company_tier" for f in resp.preferred_filters)


def test_alternative_group_only_admits_candidates_matching_a_branch():
    """A genuinely cross-field "either X-set or Y-set" requirement (see
    AlternativeGroup) -- only candidates satisfying at least one WHOLE
    branch should be returned."""
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[Filter(field="skill", operator="contains", value="Python")],
        alternative_groups=[AlternativeGroup(branches=[
            [Filter(field="education", operator="gte", value="Master")],
            [Filter(field="experience", operator="gte", value=15)],
        ])],
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "Python devs with either a Master's or 15+ years", job_id=JOB, session_id="s1",
    )
    for c in resp.candidates:
        from app.core.vocabulary import education_rank
        qualifies = (
            (education_rank(c.get("education")) or 0) >= 4
            or (c.get("experience") or 0) >= 15
        )
        assert qualifies, f"{c.get('name')} matched neither alternative branch"


def test_hard_filter_duplicate_of_alternative_branch_is_stripped():
    """Confirmed live: given "either a Master's from a Tier-1 school or 10+
    years", the model can correctly build the alternative_groups structure
    AND, separately, ALSO leave hard duplicates of the same branch filters
    in `filters` -- which forces both alternatives to be required anyway,
    recreating exactly the bug alternative_groups exists to prevent. The
    service layer must strip those hard duplicates, keeping only the
    alternative_groups copy."""
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[
            Filter(field="skill", operator="contains", value="Python"),
            Filter(field="education", operator="gte", value="Master"),  # duplicate
            Filter(field="experience", operator="gte", value=10),  # duplicate
        ],
        alternative_groups=[AlternativeGroup(branches=[
            [Filter(field="education", operator="gte", value="Master")],
            [Filter(field="experience", operator="gte", value=10)],
        ])],
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "Python devs with either a Master's or 10+ years", job_id=JOB, session_id="s1",
    )
    hard_fields = {f.field for f in resp.filters}
    assert "education" not in hard_fields
    assert "experience" not in hard_fields
    assert len(resp.alternative_groups) == 1


def test_coordinated_job_title_phrase_expands_into_alternatives():
    """Confirmed live: "senior or lead backend engineer" was emitted as ONE
    literal "contains" filter with the whole elided phrase mashed together,
    requiring a resume to contain that exact unlikely string verbatim and
    matching nobody. The service layer must expand it into a proper
    same-field "in" alternative, same mechanism as rule 3c uses for fully
    spelled-out alternatives ("Mumbai, Pune, or Bangalore")."""
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[Filter(field="job_title", operator="contains",
                         value="senior or lead backend engineer")],
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "senior or lead backend engineer", job_id=JOB, session_id="s1",
    )
    jt_filters = [f for f in resp.filters if f.field == "job_title"]
    assert len(jt_filters) == 1
    assert jt_filters[0].operator == "in"
    assert set(jt_filters[0].value) == {"senior backend engineer", "lead backend engineer"}


def test_already_complete_job_title_alternatives_pass_through_unchanged():
    """Two already fully-spelled-out alternatives (equal word counts, no
    elided head noun) need no expansion -- confirm the guard leaves this
    shape alone rather than mangling it."""
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[Filter(field="job_title", operator="contains",
                         value="software engineer or data scientist")],
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "software engineer or data scientist", job_id=JOB, session_id="s1",
    )
    jt_filters = [f for f in resp.filters if f.field == "job_title"]
    assert len(jt_filters) == 1
    assert jt_filters[0].operator == "in"
    assert set(jt_filters[0].value) == {"software engineer", "data scientist"}


def test_plain_job_title_without_or_is_untouched():
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[Filter(field="job_title", operator="contains", value="Backend Engineer")],
    )
    svc = make_service(out)
    resp = svc.filter_by_query("backend engineer", job_id=JOB, session_id="s1")
    jt_filters = [f for f in resp.filters if f.field == "job_title"]
    assert len(jt_filters) == 1
    assert jt_filters[0].operator == "contains"
    assert jt_filters[0].value == "Backend Engineer"


def test_preferred_filters_never_exclude_only_rank():
    """A candidate who fails a preferred_filters criterion must still
    appear in results -- "prefer" ranks, it never excludes."""
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[Filter(field="skill", operator="contains", value="Python")],
        preferred_filters=[Filter(field="location", operator="equals", value="Nowhereland")],
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "Python devs, preferably in Nowhereland", job_id=JOB, session_id="s1",
    )
    # Nobody is really in "Nowhereland" -- if preference were exclusionary
    # this would come back empty; it must not.
    assert resp.status == "ok"
    assert resp.showing > 0


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


def test_short_skill_term_skips_embedding_widening(monkeypatch):
    # Confirmed live on real data: a bare "AI" query's embedding similarity
    # scored a QA/performance-testing candidate with ZERO real AI/ML skills
    # HIGHEST of anyone in the job (0.687) -- nomic-embed-text can't
    # meaningfully embed a bare 2-3 character abbreviation, so the resulting
    # "shortlist" is noise, not a real ranking. Below
    # FilterService._MIN_SEMANTIC_TERM_LEN, the embedding step must be
    # skipped entirely (exact + curated taxonomy relations still apply).
    calls = []
    monkeypatch.setattr(
        service_module, "term_similarities",
        lambda job_id, term: (calls.append(term) or {}),
    )

    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="AI")])
    svc = make_service(out)
    svc.filter_by_query("engineers with AI skills", job_id=JOB, session_id="s1")
    assert calls == [], f"embedding similarity must not run for a short term, but was called for {calls}"


def test_not_contains_skill_violation_is_never_surfaced_as_partial_match(monkeypatch):
    """Confirmed live: a candidate who VIOLATES a "not_contains" skill
    exclusion (has PHP despite "no PHP") but also happens to exact-match a
    DIFFERENT positive skill filter (AWS) was still surfaced in the
    PARTIAL-match tier -- a violated exclusion was being counted the same
    as a merely-missed positive requirement in _fuzzy_skill_matches. "No
    PHP" must mean no PHP developer ever appears, full stop, not
    "downgrade them to a partial match" -- same principle as every other
    hard-exclusion guard in this module."""
    monkeypatch.setattr(service_module, "term_similarities", lambda job_id, term: {})
    candidates = [
        {"id": "c1", "name": "Has PHP And AWS", "skills": ["AWS", "PHP"]},
        {"id": "c2", "name": "Clean Match", "skills": ["AWS"]},
    ]
    monkeypatch.setattr(service_module, "get_matched_candidates", lambda job_id: candidates)

    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[
            Filter(field="skill", operator="contains", value="AWS"),
            Filter(field="skill", operator="not_contains", value="PHP"),
        ],
    )
    svc = make_service(out)
    resp = svc.filter_by_query("AWS engineers, no PHP", job_id=JOB, session_id="s1")
    names = {c.get("name") for c in resp.candidates}
    assert "Has PHP And AWS" not in names
    assert "Clean Match" in names


def test_longer_skill_term_still_uses_embedding_widening(monkeypatch):
    # Contrast with the short-term guard above -- a real, meaningful term
    # (>= _MIN_SEMANTIC_TERM_LEN) must still go through the normal
    # embedding-widening path, unaffected by that guard.
    calls = []
    monkeypatch.setattr(
        service_module, "term_similarities",
        lambda job_id, term: (calls.append(term) or {}),
    )

    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="Kubernetes")])
    svc = make_service(out)
    svc.filter_by_query("engineers with Kubernetes skills", job_id=JOB, session_id="s1")
    assert calls == ["Kubernetes"]


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


def test_combine_confirmation_still_fires_when_old_field_is_truly_untouched():
    """The genuinely ambiguous case _filter_survived_unchanged is meant to
    keep catching: an existing filter's VALUE is completely untouched, and
    a new field gets silently introduced alongside it -- still needs a
    confirmation (see PendingCombine's docstring). Uses non-skill fields
    so this runs fast and deterministic, unlike the equivalent skill-based
    test_session_state_persists_and_merges above."""
    store = InMemorySessionStore()
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals", value="Mumbai")])),
        store=store,
    )
    r1 = svc1.filter_by_query("candidates in mumbai", job_id=JOB, session_id="s1")
    assert r1.status == "ok"

    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="college_tier", operator="gte", value="High")])),
        store=store,
    )
    r2 = svc2.filter_by_query("high tier college", job_id=JOB, session_id="s1")
    assert r2.status == "clarify"
    assert r2.options == ["Yes", "No"]


def test_combine_confirmation_does_not_fire_when_old_field_value_was_replaced():
    """Regression for a confirmed live bug: a fresh, complete, self-
    contained request that happens to touch the SAME FIELD as an active
    filter but with a DIFFERENT value (a real replace, e.g. "software
    engineer" replacing "backend engineer") must apply directly -- no
    confirmation, even though it also introduces new fields alongside the
    replacement. Confirmed live: "show me software engineers ... using
    java and spring boot" (replace_all=True, correctly dropping the old
    "Backend Engineer" job_title) still triggered "Do you want candidates
    matching Software Engineer and ecommerce and Java and Spring Boot?" --
    listing nothing but what the recruiter had just explicitly said, purely
    because "job_title" as a FIELD NAME still appeared in the new result
    too. Real chat assistants don't ask you to re-approve a sentence you
    already typed in full."""
    store = InMemorySessionStore()
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="job_title", operator="contains", value="Backend Engineer")])),
        store=store,
    )
    r1 = svc1.filter_by_query("backend engineer", job_id=JOB, session_id="s1")
    assert r1.status in ("ok", "no_match")  # either way, the filter is now active in session state

    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(
            intent="FILTER_CANDIDATES", logic="AND", replace_all=True,
            filters=[
                Filter(field="job_title", operator="contains", value="Software Engineer"),
                Filter(field="college_tier", operator="gte", value="High"),
            ],
        )),
        store=store,
    )
    r2 = svc2.filter_by_query(
        "show me software engineers from a high tier college", job_id=JOB, session_id="s1",
    )
    assert r2.status != "clarify", "must not ask to confirm a complete, self-contained new request"
    fields = {f.field for f in r2.filters}
    assert "job_title" in fields and "college_tier" in fields
    assert all(f.value != "Backend Engineer" for f in r2.filters if f.field == "job_title")


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


def test_experience_search_applies_filters_from_the_same_turn(monkeypatch):
    """Regression for the confirmed live bug (see schemas.py's
    experience_query docstring): "backend engineer who worked on X" used to
    silently drop "backend engineer" entirely and search X alone, matching
    non-engineers too. A single EXPERIENCE_SEARCH turn that ALSO carries
    `filters` must apply them the same way a FILTER_CANDIDATES turn would --
    narrowing the pool BEFORE the semantic search intersects on top, in one
    turn, not requiring a separate prior turn to set the structured filter
    first (that's the older, still-supported two-turn case covered by
    test_experience_search_intersects_with_active_structured_filter above).

    Real job_title data: Ganesh B Shelke ('Senior Research Engineer' among
    his titles) matches job_title contains "Research"; Manohar Patil
    ('Team Member', 'Scientist') does not -- so a filter that both
    semantically match must still exclude Manohar."""
    class _FakePaths:
        def exists(self):
            return True
    monkeypatch.setattr(service_module.experience_index, "IndexPaths", _FakePaths)

    def fake_search(query, top_k=200):
        # Both real candidates match semantically...
        return [
            {"candidate_id": _REAL_ID_1, "score": 0.90},  # Manohar Patil
            {"candidate_id": _REAL_ID_2, "score": 0.85},  # Ganesh B Shelke
        ]
    monkeypatch.setattr(service_module.experience_index, "search", fake_search)

    out = LLMOutput(
        intent="EXPERIENCE_SEARCH",
        filters=[Filter(field="job_title", operator="contains", value="Research")],
        experience_query="led a team",
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "research engineers who led a team", job_id=JOB, session_id="s1",
    )
    ids = {c["id"] for c in resp.candidates}
    assert _REAL_ID_2 in ids, "Ganesh (real Research title) should match"
    assert _REAL_ID_1 not in ids, "Manohar (no Research title) must be excluded, not just ranked lower"

    # The structured filter must show up as a real chip too, not just
    # silently narrow the pool -- the recruiter should see WHY.
    assert any(c.field == "job_title" for c in resp.chips)
    assert any(f.field == "job_title" and f.value == "Research" for f in resp.filters)


# --------------------------------------------------------------------------- #
# _fix_missed_or_logic -- regression for a confirmed live eval failure: a
# query byte-identical to prompt.py's own "either AWS or Azure" worked
# example still came back with logic "AND" instead of "OR".
# --------------------------------------------------------------------------- #
def test_fix_missed_or_logic_flips_standalone_same_field_either_or():
    filters = [
        Filter(field="skill", operator="contains", value="AWS"),
        Filter(field="skill", operator="contains", value="Azure"),
    ]
    fixed = service_module._fix_missed_or_logic(
        "Candidates who have either AWS or Azure.", "AND", filters,
    )
    assert fixed == "OR"


def test_fix_missed_or_logic_leaves_correct_or_alone():
    filters = [
        Filter(field="skill", operator="contains", value="AWS"),
        Filter(field="skill", operator="contains", value="Azure"),
    ]
    fixed = service_module._fix_missed_or_logic(
        "Candidates who have either AWS or Azure.", "OR", filters,
    )
    assert fixed == "OR"


def test_fix_missed_or_logic_does_not_fire_without_either_or_phrasing():
    filters = [
        Filter(field="skill", operator="contains", value="AWS"),
        Filter(field="skill", operator="contains", value="Azure"),
    ]
    fixed = service_module._fix_missed_or_logic("knows AWS and Azure", "AND", filters)
    assert fixed == "AND"


def test_fix_missed_or_logic_does_not_fire_when_either_or_is_nested_in_a_bigger_query():
    # "Python developers in Mumbai with either Kubernetes or Terraform" --
    # the either/or here is expressed via alternative_groups, not `filters`,
    # so the two DIFFERENT-field filters that DO reach this check (skill,
    # location) must never get flipped to OR (that would wrongly make
    # Python optional too).
    filters = [
        Filter(field="skill", operator="contains", value="Python"),
        Filter(field="location", operator="equals", value="Mumbai"),
    ]
    fixed = service_module._fix_missed_or_logic(
        "Python developers in Mumbai with either Kubernetes or Terraform experience.",
        "AND", filters,
    )
    assert fixed == "AND"


def test_or_logic_fix_applies_end_to_end():
    out = LLMOutput(
        intent="FILTER_CANDIDATES", logic="AND",
        filters=[
            Filter(field="skill", operator="contains", value="AWS"),
            Filter(field="skill", operator="contains", value="Azure"),
        ],
    )
    svc = make_service(out)
    resp = svc.filter_by_query(
        "Candidates who have either AWS or Azure.", job_id=JOB, session_id="s1",
    )
    assert resp.logic == "OR"


# --------------------------------------------------------------------------- #
# _is_untracked_lookup -- regression for a confirmed live eval failure: a
# query byte-identical to prompt.py's own "What's his email address?"
# worked example still came back as intent LOOKUP instead of
# UNSUPPORTED_FILTER.
# --------------------------------------------------------------------------- #
def test_is_untracked_lookup_matches_contact_detail_questions():
    for q in ("what's his email address", "What is her phone number?",
              "can I get his contact details", "do you have a resume", "his LinkedIn"):
        assert service_module._is_untracked_lookup(q), q


def test_is_untracked_lookup_does_not_match_real_fields():
    for q in ("which college did he go to", "what's her notice period",
              "where did this candidate work"):
        assert not service_module._is_untracked_lookup(q), q


def test_untracked_lookup_override_applies_end_to_end():
    # A FakeLLM that (wrongly, matching the confirmed live failure) still
    # returns LOOKUP for this -- the deterministic override must catch it
    # before _answer_lookup ever runs, regardless of what the model said.
    svc = make_service(LLMOutput(intent="LOOKUP", lookup_field="email"))
    resp = svc.filter_by_query(
        "what's his email address?", job_id=JOB, session_id="s1",
    )
    assert resp.status == "unsupported"
    assert "contact" in resp.message.lower()
