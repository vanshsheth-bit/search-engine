"""Service-layer tests using a fake LLM (no Ollama needed)."""
from __future__ import annotations

import app.core.service as service_module
from app.core.service import FilterService
from app.core.session import InMemorySessionStore
from app.models.schemas import AlternativeGroup, FilterSpec, LLMOutput, Filter, SessionState

# A real jdId from the real matched-candidates dataset (111 candidates
# after dedup-by-real-person -- see _identity_key in candidates.py --
# includes real Mumbai/Python candidates). See app/core/candidates.py.
JOB = "6a8c26ee15f64740b81997da"

# Real job with real DevOps-classified experience entries (see
# candidates._load_candidate_domain_years): Adam Hardesty 0.7y, Andrew Zhang
# 1.2y, Anthony Knight 2.2y, Antoine Fongang 2.3y, Art Frick 1.1y -- used
# specifically for domain_experience tests, since JOB above has none.
JOB_DEVOPS = "00000103"

# _fuzzy_skill_matches (app/core/service.py) used to also call out to Ollama
# for an LLM-verified "semantic similarity" tier, which needed pinning inert
# here to keep this file's tests deterministic and Ollama-free. That tier
# was REMOVED entirely (confirmed live: non-functional with thinking off --
# rubber-stamped literally any candidate as a match regardless of the skill
# asked -- and 15+ minutes for a realistic shortlist with thinking on, not
# viable for an interactive search; see service.py's comment on
# _fuzzy_skill_matches). Only the deterministic, local, no-network
# taxonomy-related-tool path remains, so no monkeypatching is needed
# anymore for this file to stay exactly what its docstring already claims.


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


def patch_pool(monkeypatch, candidates: list[dict]) -> None:
    """Point the service at a synthetic candidate pool. An `experience_text`
    key written on a candidate here is lifted into the SIDE INDEX the real
    pipeline reads (candidates.experience_texts_by_candidate) rather than
    left on the candidate dict -- see that function's docstring for why it
    deliberately isn't candidate data (it leaked into API responses and
    session state when it was)."""
    texts = {c["id"]: c["experience_text"] for c in candidates if c.get("experience_text")}
    pool = [{k: v for k, v in c.items() if k != "experience_text"} for c in candidates]
    monkeypatch.setattr(service_module, "get_matched_candidates", lambda job_id: pool)
    monkeypatch.setattr(service_module, "experience_texts_by_candidate", lambda: texts)


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
    # A real, populated field (total career "experience") with an absurd
    # threshold nobody could ever satisfy -- unlike skill_experience (see
    # test_skill_experience_falls_back_to_plain_skill_when_no_years_data),
    # this field is genuinely present on every candidate, so "no_match" here
    # means a real, meaningful zero, not a structural data gap.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="experience", operator="gte", value=99)])
    svc = make_service(out)
    resp = svc.filter_by_query("99 years experience", job_id=JOB, session_id="s1")
    assert resp.status == "no_match"
    assert resp.showing == 0
    assert resp.suggestions


def test_skill_experience_falls_back_to_plain_skill_when_no_years_data():
    # candidates._skill_years_from_experience now computes a REAL per-skill
    # years number for known tools where a resume's own job-description text
    # names them (confirmed dataset-wide: 65% of candidates who list Python
    # have it named in at least one of their own job descriptions) -- but
    # NOT for every skill/pool combination. On THIS job's specific 111-
    # candidate pool, none of the 8 real Python matches happens to have it
    # in their own description text, so it's still unanswerable HERE (see
    # _skill_years_available, which is checked per-skill-per-pool, never
    # pool-wide-for-any-skill, precisely so a real number computed for a
    # DIFFERENT skill/candidate elsewhere can't wrongly vouch for this one
    # and turn a genuine data gap into a silent, wrong no_match). The fix:
    # detect the gap against the real pool for the SPECIFIC named skill and
    # fall back to a plain "has this skill" filter, telling the recruiter
    # why instead of a silent, permanent no_match.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill_experience", operator="gte",
                                    skill="Python", value=3)])
    svc = make_service(out)
    resp = svc.filter_by_query("3 years of python", job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    # Same 8 real matches as a plain "skill contains python" query (see
    # test_fuzzy_full_match_is_tagged_not_indistinguishable_from_exact) --
    # the years constraint was dropped, not silently ignored.
    assert resp.showing == 8
    assert resp.filters == [Filter(field="skill", operator="contains", value="Python")]
    assert "doesn't reliably track years" in resp.message
    assert "Python" in resp.message


def test_skill_experience_with_untracked_skill_degrades_to_flat_experience():
    # Real, reported live bug (found via a 110-query domain sweep): when the
    # skill_experience filter's named "skill" isn't a real tool or subdomain
    # at all -- an activity/practice phrase like "leading operations teams"
    # or "corporate legal", not a trackable named skill -- it used to hit
    # the SAME no-years-data fallback as a real-but-undated tool (see
    # test_skill_experience_falls_back_to_plain_skill_when_no_years_data),
    # degrading to a literal `skill contains "leading operations teams"`
    # filter that's close to meaningless (nobody's resume names a fake
    # skill) while silently dropping the recruiter's explicit number. Since
    # the term is untracked (_is_untracked_term), fall back to a flat
    # pool-wide `experience` filter instead, preserving the number.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill_experience", operator="gte",
                                    skill="leading operations teams", value=12)])
    svc = make_service(out)
    resp = svc.filter_by_query(
        "at least 12 years leading operations teams", job_id=JOB, session_id="s1",
    )
    assert resp.status == "ok"
    assert resp.filters == [Filter(field="experience", operator="gte", value=12)]
    assert "isn't specific enough to scope a years-of-experience filter" in resp.message
    assert "leading operations teams" in resp.message


def test_incomplete_skill_experience_is_repaired_before_the_no_years_data_fallback():
    # Real, reproduced live (3 phrasings, originally with "DevOps" -- see
    # test_skill_matching_a_position_reclassifies_to_domain for why a
    # REAL tool name is used here instead, so this test isolates just the
    # missing-skill-field repair without also triggering that separate
    # domain-reclassification feature): qwen3:4b correctly picks the
    # skill_experience FIELD but leaves its `skill` sub-field empty, while
    # ALSO emitting a redundant separate skill filter for the same term.
    # Without _repair_incomplete_skill_experience (service.py),
    # validation.py drops the skill-less skill_experience filter entirely
    # (nothing to check it against) and the years number is silently lost,
    # leaving only the plain skill filter with a confusing "I need to know
    # which skill..." message -- even though the recruiter named the skill
    # right there in the same query. The repair merges them into one
    # correct skill_experience filter first, which then hits the SAME
    # no-years-data fallback as test_skill_experience_falls_back_to_
    # plain_skill_when_no_years_data.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[
                        Filter(field="skill_experience", operator="gte", value=2),
                        Filter(field="skill", operator="contains", value="kubernetes"),
                    ])
    svc = make_service(out)
    resp = svc.filter_by_query("2+ years of kubernetes experience", job_id=JOB, session_id="s1")
    assert resp.status in ("ok", "no_match")  # real data-dependent count, not asserted here
    assert resp.filters == [Filter(field="skill", operator="contains", value="Kubernetes")]
    assert "doesn't reliably track years" in resp.message


def test_skill_experience_resolves_for_real_when_the_resume_text_supports_it():
    # Positive case for candidates._skill_years_from_experience: unlike JOB
    # above (where none of the real Python matches happen to name it in
    # their own job-description text), JOB_DEVOPS's pool has real candidates
    # who DO -- so this skill_experience filter must resolve for real, not
    # degrade to the plain-skill fallback (contrast with
    # test_skill_experience_falls_back_to_plain_skill_when_no_years_data,
    # same skill, different pool). Real data: 23 of JOB_DEVOPS's 99
    # candidates have a real computed Python-years number; 13 of those are
    # >= 3.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill_experience", operator="gte",
                                    skill="Python", value=3)])
    svc = make_service(out)
    resp = svc.filter_by_query("3 years of python", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "ok"
    assert resp.showing == 13
    # The real filter survives untouched -- no fallback message, no
    # degradation to a plain "has this skill" filter.
    assert resp.filters == [Filter(field="skill_experience", operator="gte",
                                    value=3, skill="Python")]
    assert resp.message is None
    names = {c["name"] for c in resp.candidates}
    assert "Abhinav Srivastava" in names  # real: 3.6 computed years of Python
    assert "Andrew Zhang" in names  # real: >= 3 computed years of Python
    # A candidate whose real computed Python years are below the threshold
    # must NOT be silently included by falling back to plain presence.
    assert "Akhila Makthala" not in names  # real: 0.5 computed years of Python
    # Card-level annotation (see service._annotate_skill_match_years): the
    # real per-skill number is surfaced alongside (not replacing) total
    # career years, same principle as domain_match_years.
    abhinav = next(c for c in resp.candidates if c["name"] == "Abhinav Srivastava")
    assert abhinav["skill_match_years"] == [{"skill": "Python", "years": 3.6}]


def test_skill_match_years_annotates_a_plain_skill_search_too():
    # skill_match_years isn't limited to skill_experience filters -- a plain
    # "python" search (no years threshold, the common case: "give me python
    # guy") gets the same annotation whenever a real number exists, which is
    # exactly the real, reported confusion this closes (a card showing only
    # total career years next to "Found N candidates matching Python" reads
    # as if that total WERE their Python experience).
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="python")])
    svc = make_service(out)
    resp = svc.filter_by_query("python", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "ok"
    abhinav = next(c for c in resp.candidates if c["name"] == "Abhinav Srivastava")
    assert abhinav["skill_match_years"] == [{"skill": "Python", "years": 3.6}]
    # A real match with NO computed number for Python must not get a
    # fabricated or borrowed annotation -- absent, not zero/wrong.
    others_without_tag = [c for c in resp.candidates if "skill_match_years" not in c]
    assert others_without_tag  # most real matches still have no real number


def test_skill_matching_a_position_reclassifies_to_domain_experience():
    # Real, reported gap: "3 years of exp in DevOps" was treated as a
    # skill_experience filter with skill="DevOps" -- but DevOps isn't a
    # tool anyone "has" (confirmed absent from the entire ~16,800-tool
    # taxonomy, see skill_taxonomy.is_known_tool); it's a practice area a
    # person works IN, matching one of the 212 real categories the
    # experience classifier uses to tag a candidate's ACTUAL work history.
    # Fixed: reclassify to `domain_experience`, NOT a bare `domain`
    # contains -- the years are preserved and actually verified against
    # real per-experience durations summed by subdomain (see
    # candidates._load_candidate_domain_years), not dropped. Real data on
    # JOB_DEVOPS: Adam Hardesty 0.7y, Andrew Zhang 1.2y, Anthony Knight
    # 2.2y, Antoine Fongang 2.3y, Art Frick 1.1y of DevOps-classified
    # experience -- only the two >= 2y actually qualify, a genuinely
    # different (and correct) result from "ever did any DevOps work" or
    # "has N years of TOTAL career experience".
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill_experience", operator="gte",
                                    skill="DevOps", value=2)])
    svc = make_service(out)
    resp = svc.filter_by_query("2 years of exp in devops", job_id=JOB_DEVOPS, session_id="s1")
    # Real taxonomy tools + real candidates in JOB_DEVOPS having them means
    # this now lands on the domain_skill_pick step (see
    # test_domain_only_query_offers_real_skill_choices) rather than
    # searching immediately -- reclassification to domain_experience still
    # happened underneath, just carried in `domain_filter` instead of
    # `filters` until the recruiter finishes (or skips) that step.
    assert resp.status == "domain_skill_pick"
    assert resp.domain_filter == Filter(field="domain_experience", operator="gte",
                                        value=2, skill="DevOps")
    assert "practice area" in resp.message
    assert "DevOps" in resp.message

    # Searching with NONE checked (the "just show me everyone" skip path)
    # applies domain_filter exactly as-is -- confirms the underlying
    # domain_experience filtering logic is unaffected by the new step.
    from app.models.schemas import PatchStateRequest
    apply_resp = svc.patch_state(PatchStateRequest(
        job_id=JOB_DEVOPS, session_id="s1", filters=[resp.domain_filter], logic="AND",
    ))
    names = {c["name"] for c in apply_resp.candidates}
    assert names == {"Anthony Knight", "Antoine Fongang"}

    # A genuine tool name is NEVER reclassified this way, even if it were
    # to (hypothetically) collide with a subdomain label -- a real tool
    # match always wins. Confirmed here with a real tool the taxonomy does
    # recognize.
    out2 = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="Python")])
    svc2 = make_service(out2)
    resp2 = svc2.filter_by_query("python", job_id=JOB, session_id="s2")
    assert resp2.filters == [Filter(field="skill", operator="contains", value="Python")]


def test_typo_tier_company_reclassifies_instead_of_literal_name_search():
    # Real, reported live bug: "give me guy who has worked in high tire
    # company" ("tier" mistyped as "tire") was NOT recognized as a
    # company-tier request -- the model fell back to a literal company
    # NAME search (`company contains "high tire"`), which can never match
    # anyone (no company is named "high tire"). Must reclassify to the
    # real ordinal company_tier filter instead.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="company", operator="contains", value="high tire")])
    svc = make_service(out)
    resp = svc.filter_by_query(
        "give me guy who has worked in high tire company", job_id=JOB, session_id="s1",
    )
    assert resp.filters == [Filter(field="company_tier", operator="gte", value="High")]
    assert "high tire" in resp.message and "High tier" in resp.message


def test_typo_tier_university_reclassifies_too():
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="university", operator="contains", value="medium tyre")])
    svc = make_service(out)
    resp = svc.filter_by_query("studied at a medium tyre school", job_id=JOB, session_id="s1")
    assert resp.filters == [Filter(field="college_tier", operator="gte", value="Medium")]


def test_typo_tier_landing_on_bare_skill_field_reclassifies_too():
    # Real, reported live bug: "give me guy with python exprence in high
    # tire company" (already typo-corrected to "...high tier company" by
    # _correct_known_query_typos before the LLM ever sees it) STILL
    # sometimes made the model emit {"field":"skill","value":"high tier"}
    # instead of company_tier -- no company/university field on the filter
    # itself to catch via _TIER_FIELD_MAP, so this is a separate path in
    # _reclassify_typo_tier_filter keyed off the query's own wording
    # (no "college"/"university"/"school" mention here -> company_tier).
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="Python"),
                             Filter(field="skill", operator="contains", value="high tier")])
    svc = make_service(out)
    resp = svc.filter_by_query(
        "give me guy with python exprence in high tier company", job_id=JOB, session_id="s1",
    )
    assert Filter(field="skill", operator="contains", value="Python") in resp.filters
    assert Filter(field="company_tier", operator="gte", value="High") in resp.filters
    assert not any(f.field == "skill" and f.value == "high tier" for f in resp.filters)
    assert "high tier" in resp.message and "High tier" in resp.message


def test_typo_tier_landing_on_bare_skill_field_prefers_college_tier_in_college_context():
    # Same misreading, but the query's own wording is about a COLLEGE, not
    # a company -- must disambiguate to college_tier, not default blindly
    # to company_tier.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="high tier")])
    svc = make_service(out)
    resp = svc.filter_by_query(
        "someone who studied at a high tier college", job_id=JOB, session_id="s1",
    )
    assert resp.filters == [Filter(field="college_tier", operator="gte", value="High")]


def test_a_real_company_name_that_happens_to_contain_tier_words_is_untouched():
    # Only an EXACT "<qualifier> <tier typo>" value reclassifies -- a real
    # company name must never be silently rewritten just because it
    # happens to contain similar-looking words elsewhere in a longer name.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="company", operator="contains", value="High Tier Capital Partners")])
    svc = make_service(out)
    resp = svc.filter_by_query("worked at High Tier Capital Partners", job_id=JOB, session_id="s1")
    assert resp.filters == [Filter(field="company", operator="contains", value="High Tier Capital Partners")]


def test_correct_known_query_typos_fixes_the_compound_typo():
    # The actual originally-reported live bug: TWO typos at once ("tire"
    # for "tier" AND "collage" for "college") made the model extract
    # NOTHING at all -- confirmed live with no history: structured=[],
    # tools=[]. Correctly-spelled "high tier college" resolves fine, so
    # the fix is upstream of the LLM entirely: correct the query TEXT
    # before it ever reaches the model.
    assert service_module._correct_known_query_typos(
        "high tire collage",
    ) == "high tier college"
    assert service_module._correct_known_query_typos(
        "give me guy who has worked in high tire company",
    ) == "give me guy who has worked in high tier company"
    assert service_module._correct_known_query_typos(
        "low tyre college",
    ) == "low tier college"
    assert service_module._correct_known_query_typos(
        "studied at a good univercity",
    ) == "studied at a good university"


def test_correct_known_query_typos_never_touches_a_bare_unqualified_tire():
    # "tire" has a real, common, unrelated meaning (the rubber wheel
    # component) -- must NOT be corrected without a tier-qualifier word
    # (low/medium/high) immediately before it, or a legitimate query about
    # the tire/automotive industry would be silently corrupted.
    query = "candidate with tire manufacturing experience"
    assert service_module._correct_known_query_typos(query) == query


def test_compound_tier_college_typo_resolves_end_to_end():
    # Confirms the correction actually reaches the LLM (not just the
    # standalone function in isolation) -- a fake that records what query
    # it was called with, and that the request resolves correctly once
    # the LLM sees the corrected text.
    received = {}

    class RecordingFakeLLM:
        def translate(self, query, current_filters, history=None):
            received["query"] = query
            return LLMOutput(
                intent="FILTER_CANDIDATES", logic="AND",
                filters=[Filter(field="college_tier", operator="gte", value="High")],
            )

    svc = FilterService(llm=RecordingFakeLLM(), store=InMemorySessionStore())
    resp = svc.filter_by_query("high tire collage", job_id=JOB, session_id="s1")
    assert received["query"] == "high tier college"
    assert resp.filters == [Filter(field="college_tier", operator="gte", value="High")]


def test_untracked_multiword_phrase_gets_an_honest_note_not_silence():
    # Real, reported live bug: a multi-word term absent from BOTH the tool
    # taxonomy and all 207 real classified subdomains (nothing to widen or
    # reclassify it into) used to silently become a literal,
    # nobody-has-this-exact-phrase search with zero explanation. Uses a
    # nonsense phrase (confirmed absent from JOB_DEVOPS's real experience
    # text, unlike "supply chain platform" -- see
    # test_untracked_domain_phrase_now_resolves_via_real_experience_text_on_the_real_dataset
    # below, which the ORIGINAL literal reported bug now resolves through)
    # so this test still exercises the genuine "found nothing anywhere,
    # including via the experience-text fallback" case and its note.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains",
                                    value="xylophone manufacturing pipeline")])
    svc = make_service(out)
    resp = svc.filter_by_query("worked on xylophone manufacturing pipeline", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "no_match"
    assert "xylophone manufacturing pipeline" in resp.message
    assert "isn't a specific skill or practice area" in resp.message

    # A genuinely untracked SINGLE-word term stays silent -- common and not
    # worth an apologetic note (the taxonomy doesn't claim to know every
    # real tool name).
    out2 = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                     filters=[Filter(field="skill", operator="contains",
                                     value="some-made-up-tool-xyz")])
    svc2 = make_service(out2)
    resp2 = svc2.filter_by_query("knows some-made-up-tool-xyz", job_id=JOB_DEVOPS, session_id="s2")
    assert resp2.message is None or "isn't a specific skill" not in (resp2.message or "")


def test_untracked_domain_field_emitted_directly_also_gets_an_honest_note():
    # Real, reported live bug: the honest "isn't tracked" note above only
    # ever fired for a term that started life as `skill`/`skill_experience`
    # and got reclassified into domain -- a `domain` filter the LLM emitted
    # DIRECTLY (no reclassification needed, it's already the right field)
    # got no equivalent check at all, so a genuinely untracked domain term
    # (confirmed absent from every real subdomain, substring or otherwise --
    # unlike "B2B sales", which turned out to genuinely substring-match
    # "enterprise & b2b sales") went completely silent on `no_match`
    # instead of explaining why.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="domain", operator="contains",
                                    value="xylophone manufacturing pipeline")])
    svc = make_service(out)
    resp = svc.filter_by_query(
        "who has worked in xylophone manufacturing pipeline", job_id=JOB_DEVOPS, session_id="s1",
    )
    assert resp.status == "no_match"
    assert "xylophone manufacturing pipeline" in resp.message
    assert "isn't a specific skill or practice area" in resp.message


def test_domain_field_that_substring_matches_a_real_subdomain_stays_silent():
    # The other half of the same fix: "sales operations" is a genuine
    # SUBSTRING of the real classified subdomain "Revenue Operations &
    # Sales Operations" -- domain filters match by substring at apply time
    # (see _annotate_domain_match_years), so this must NOT be flagged as
    # untracked just because it isn't an EXACT subdomain name. An exact-
    # match check here would have wrongly apologized for a term that
    # actually works fine.
    filters = [Filter(field="domain", operator="contains", value="sales operations")]
    out, note = service_module._reclassify_skill_as_domain_when_its_a_position(filters)
    assert out == filters
    assert note is None


def test_untracked_domain_phrase_now_resolves_via_real_experience_text_on_the_real_dataset():
    # The actual originally-reported live bug, end to end, on the REAL
    # dataset (no monkeypatching): "give me a guy who has worked on supply
    # chain platform" resolved to a `domain contains "supply chain
    # platform"` filter with nothing to match against (confirmed absent
    # from both the tool taxonomy and the 207 subdomains) -- previously
    # always "no_match". Real candidates on this job (Shweta Soam,
    # Mohd Hasnain Sayed, Sobhan Chatterjee, ...) DO describe supply-chain
    # work in their own real job-history text (e.g. "Supply Chain
    # Optimization", "supply chain robustness") -- just never with the
    # literal trailing word "platform" the recruiter happened to type,
    # which is exactly why _strip_domain_noise_suffix exists (see its
    # docstring): the generic artifact-type word ("platform"/"system"/...)
    # is stripped so the real concept ("supply chain") is what's searched
    # for, not the recruiter's exact 3-word phrasing.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="domain", operator="contains",
                                    value="supply chain platform")])
    svc = make_service(out)
    resp = svc.filter_by_query("worked on supply chain platform", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "ok"
    assert resp.showing > 0
    assert all(c.get("experience_text_match") for c in resp.candidates)
    assert "not a tracked category" in resp.message


def test_untracked_phrase_found_via_experience_text_search_surfaces_with_a_note(monkeypatch):
    # The follow-up fix for the same "supply chain platform" gap above:
    # instead of only a literal, nobody-has-this-exact-phrase search against
    # the (nonexistent) tracked field, also search every candidate's real
    # job-history text (see candidates._adapt_resume's `experience_text`)
    # for a literal mention of the untracked term -- same deterministic
    # keyword-search idea as candidates._skill_years_from_experience, just
    # over an arbitrary phrase instead of one of a candidate's own declared
    # skills. Synthetic candidates (not the real dataset) so the match is
    # deterministic and doesn't depend on what happens to be in real resumes.
    fake_candidates = [
        {"id": "c1", "name": "Match Guy", "skills": {"Python": {"years": None}},
         "experience_text": "Led development of a supply chain platform for logistics."},
        {"id": "c2", "name": "No Match Guy", "skills": {"Python": {"years": None}},
         "experience_text": "Worked on an unrelated billing system."},
    ]
    patch_pool(monkeypatch, fake_candidates)
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains",
                                    value="supply chain platform")])
    svc = make_service(out)
    resp = svc.filter_by_query("worked on supply chain platform", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "ok"
    assert resp.showing == 1
    assert resp.candidates[0]["name"] == "Match Guy"
    assert resp.candidates[0]["experience_text_match"] == [{"term": "supply chain platform"}]
    assert "supply chain platform" in resp.message
    assert "not a tracked category" in resp.message


def test_experience_text_search_does_not_reduce_a_2word_phrase_to_a_bare_common_word(monkeypatch):
    # Real, reported live bug: "payment system" (2 words) stripped its
    # trailing noise word ("system") down to the single bare word
    # "payment" -- which then matched anyone whose resume mentions payment
    # ANYTHING (payment methods, "Payment Card Industry" audits, vendor
    # payment issues, cash payment...), none of whom described building a
    # payment system. _strip_domain_noise_suffix now refuses to strip a
    # 2-word phrase down to one word (requires 3+ words before stripping),
    # so a bare-word mention must NOT count as a match, while the real
    # phrase (singular OR plural, see _phrase_patterns) still does.
    fake_candidates = [
        {"id": "c1", "name": "Bare Word Guy", "skills": {},
         "experience_text": "Handled PCI (Payment Card Industry) audits and vendor payment issues."},
        {"id": "c2", "name": "Real Match Guy", "skills": {},
         "experience_text": "Designed and built the core payment systems for the platform."},
    ]
    patch_pool(monkeypatch, fake_candidates)
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains",
                                    value="payment system")])
    svc = make_service(out)
    resp = svc.filter_by_query("worked on payment system", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "ok"
    assert resp.showing == 1
    assert resp.candidates[0]["name"] == "Real Match Guy"


def test_experience_text_search_skips_a_spurious_single_word_untracked_skill(monkeypatch):
    # Real, reported live bug (original form): a compound query ("mid-level
    # backend engineer familiar with Node.js or Django...") caused the model
    # to ALSO emit a spurious `skill contains "backend"` filter -- a bare,
    # common English word, already covered by the same query's own
    # `job_title contains "backend engineer"`. The risk this guards against
    # is a FREE-TEXT substring search matching "backend" against nearly any
    # engineering resume's prose (same false-positive class as "payment"-
    # from-"payment system") -- this fake candidate's experience_text says
    # "Tested backend APIs" but they are NOT a backend engineer.
    #
    # UPDATED, separately-confirmed-live bug fixed the same session as this
    # test's assertion below: "backend" (and other bare practice-area words)
    # is not free-text-searched at all any more -- it's reclassified into a
    # structured `domain contains "backend"` filter (see
    # _reclassify_skill_as_domain_when_its_a_position; 0 real candidates on
    # job 00000103 have "backend" as a literal skill tag, so this is safe --
    # contrast test_results_rank_exact_before_fuzzy_full_before_partial's
    # "Agile", which genuinely IS a real literal skill tag and is correctly
    # excluded from this same widening via _real_literal_skill_terms).
    # This fake candidate has no `domain` data at all, so the ORIGINAL
    # protection still holds -- they correctly get ZERO matches -- but now
    # via structured-field absence, not a free-text miss, and WITH an honest
    # explanation instead of silence.
    fake_candidates = [
        {"id": "c1", "name": "Unrelated Backend Mention",
         "skills": {}, "job_title": ["QA Analyst"],
         "experience_text": "Tested backend APIs for a mobile app."},
    ]
    patch_pool(monkeypatch, fake_candidates)
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="backend")])
    svc = make_service(out)
    resp = svc.filter_by_query("backend engineer", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "no_match"
    assert resp.filters == [Filter(field="domain", operator="contains",
                                    value="backend", hard=True)]


def test_payment_system_false_positive_fix_on_the_real_dataset():
    # The actual originally-reported live bug, end to end, on the REAL
    # dataset: before the 2-word-noise-strip fix above, this query
    # (mis-)matched 12 candidates on JOB_DEVOPS purely because "payment"
    # alone is a common word in unrelated prose (payment methods, PCI
    # audits, vendor payment issues...) -- Ahmed Sadig is the only real
    # candidate here who actually describes payment systems work (his own
    # text literally says "payment systems", plural -- see
    # _phrase_patterns for why the singular query still finds it).
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains",
                                    value="payment system")])
    svc = make_service(out)
    resp = svc.filter_by_query("worked on payment system", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "ok"
    names = {c["name"] for c in resp.candidates}
    assert names == {"Ahmed Sadig"}


def test_candidate_confirmed_via_text_search_is_promoted_not_duplicated(monkeypatch):
    # Real, reported live bug: a candidate missing ONLY an untracked skill
    # term (e.g. "cloud technologies", which has nothing in the taxonomy to
    # ever confirm) sat in `partial` from _fuzzy_skill_matches, tagged
    # "missing: cloud technologies" -- but the SAME candidate ALSO
    # independently qualified via _experience_text_matches, since their own
    # job-history text literally mentions it. The two widening passes don't
    # know about each other, so the person appeared TWICE in the final
    # list: once as an untagged full match, once still flagged partial and
    # incomplete. Must appear exactly ONCE, promoted to a full match (not
    # left dangling in partial once text search already confirmed it).
    fake_candidates = [
        {"id": "c1", "name": "Both Ways Guy", "skills": {"AWS": {"years": 2.0}},
         "experience_text": "Worked extensively with cloud technologies and AWS infrastructure."},
    ]
    patch_pool(monkeypatch, fake_candidates)
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND", filters=[
        Filter(field="skill", operator="contains", value="AWS"),
        Filter(field="skill", operator="contains", value="cloud technologies"),
    ])
    svc = make_service(out)
    resp = svc.filter_by_query(
        "AWS and cloud technologies", job_id=JOB_DEVOPS, session_id="s1",
    )
    assert resp.showing == 1
    assert [c["name"] for c in resp.candidates] == ["Both Ways Guy"]
    assert "partial_skill_match" not in resp.candidates[0]
    assert resp.candidates[0].get("experience_text_match") == [{"term": "cloud technologies"}]


def test_seniority_hard_band_expands_to_years_and_title_routes():
    # The actual feature: a bare "mid level" query resolves DETERMINISTICALLY
    # to a defined years range OR a job_title keyword check, rather than
    # CLARIFY-ing every time (see vocabulary.SENIORITY_BANDS).
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="seniority", operator="equals", value="mid level")])
    svc = make_service(out)
    resp = svc.filter_by_query("mid level backend engineer", job_id=JOB, session_id="s1")
    assert resp.status != "clarify"
    assert not any(f.field == "seniority" for f in resp.filters)
    assert len(resp.alternative_groups) == 2
    years_route = next(g for g in resp.alternative_groups if g.filters[0].field == "experience")
    assert {(f.operator, f.value) for f in years_route.filters} == {("gte", 3), ("lte", 7)}
    title_route = next(g for g in resp.alternative_groups if g.filters[0].field == "job_title")
    assert title_route.filters[0].operator == "in"
    assert "Mid-Level" in title_route.filters[0].value


def test_seniority_soft_preference_keeps_the_floor_not_a_colliding_pair():
    # Regression test for a real bug caught during planning: emitting BOTH
    # `experience gte 3` and `experience lte 7` as separate flat Filters
    # would collide on the same merge_filters/Filter.key() (experience isn't
    # in _MULTI_VALUE_FIELDS) and silently lose one of the two bounds. The
    # soft path must emit ONLY the floor -- this proves it survives, not
    # just that *some* experience filter exists.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="seniority", operator="equals",
                                    value="mid level", hard=False)])
    svc = make_service(out)
    resp = svc.filter_by_query("prefer mid-level candidates", job_id=JOB, session_id="s1")
    assert resp.filters == [Filter(field="experience", operator="gte", value=3, hard=False)]
    assert resp.alternative_groups == []


def test_seniority_unrecognized_value_dropped_with_explanatory_note():
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="seniority", operator="equals", value="blah level")])
    svc = make_service(out)
    resp = svc.filter_by_query("blah level engineer", job_id=JOB, session_id="s1")
    assert resp.status != "error"
    assert resp.filters == []
    assert resp.alternative_groups == []
    assert "recognize" in resp.message


def test_seniority_degrades_when_a_real_alternative_group_is_stated_same_turn():
    # FilterSpec.alternative_groups is one flat, non-nested OR-level -- it
    # cannot express "(seniority route A or B) AND (Master's or 10 years)".
    # Combining them would either wrongly let "mid level" alone satisfy the
    # WHOLE requirement, or (via merge_alternative_groups' wholesale-replace)
    # silently discard the real, unrelated either/or. Must degrade instead.
    real_groups = [
        AlternativeGroup(filters=[Filter(field="education", operator="gte", value="Master"),
                                   Filter(field="college_tier", operator="gte", value="High")]),
        AlternativeGroup(filters=[Filter(field="experience", operator="gte", value=10)]),
    ]
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="seniority", operator="equals", value="mid level")],
                    alternative_groups=real_groups)
    svc = make_service(out)
    resp = svc.filter_by_query(
        "mid level, and either a master's from a tier-1 university or 10 years experience",
        job_id=JOB, session_id="s1",
    )
    # The real either/or survives untouched -- not combined with a seniority
    # group, not replaced.
    assert len(resp.alternative_groups) == 2
    assert {f.field for g in resp.alternative_groups for f in g.filters} == \
        {"education", "college_tier", "experience"}
    # The seniority band degrades to a floor-only hard filter instead.
    assert Filter(field="experience", operator="gte", value=3, hard=True) in resp.filters


def test_seniority_alone_does_not_wholesale_replace_a_carried_over_alternative_group():
    # A DIFFERENT version of the same hazard: an EARLIER turn's real
    # alternative_groups must survive a LATER turn that only states a
    # seniority band (no new alternative_groups, replace_all=False) --
    # merge_alternative_groups wholesale-replaces whenever `incoming` is
    # non-empty, so injecting a seniority-derived group here would silently
    # discard turn 1's real requirement.
    store = InMemorySessionStore()
    real_groups = [
        AlternativeGroup(filters=[Filter(field="education", operator="gte", value="Master")]),
        AlternativeGroup(filters=[Filter(field="experience", operator="gte", value=10)]),
    ]
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    alternative_groups=real_groups)),
        store=store,
    )
    r1 = svc1.filter_by_query("either a master's or 10 years experience", job_id=JOB, session_id="s1")
    assert len(r1.alternative_groups) == 2

    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="seniority", operator="equals", value="mid level")])),
        store=store,
    )
    r2 = svc2.filter_by_query("also mid level", job_id=JOB, session_id="s1")
    assert len(r2.alternative_groups) == 2
    assert {f.field for g in r2.alternative_groups for f in g.filters} == {"education", "experience"}
    assert Filter(field="experience", operator="gte", value=3, hard=True) in r2.filters


def test_domain_only_query_offers_real_skill_choices():
    # The actual feature request: "I want a DevOps guy" is vague -- rather
    # than searching immediately (which would just mean "anyone ever
    # classified into DevOps"), offer real skill choices to narrow it,
    # grounded in what candidates in THIS job's real pool actually have
    # (not the full ~188-tool taxonomy list for "DevOps", which would be
    # overwhelming and mostly irrelevant here -- see
    # service._domain_skill_options).
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="domain", operator="contains", value="DevOps")])
    svc = make_service(out)
    resp = svc.filter_by_query("give me devops guy", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status == "domain_skill_pick"
    assert resp.domain_filter == Filter(field="domain", operator="contains", value="DevOps")
    # Real tools confirmed present among JOB_DEVOPS's real DevOps-classified
    # candidates (Docker, Ansible, Jenkins, Kubernetes, Terraform, ...) --
    # never a tool from the taxonomy nobody here actually has.
    assert resp.skill_options
    assert "Docker" in resp.skill_options
    assert all(isinstance(s, str) for s in resp.skill_options)
    # Nothing applied yet -- old (empty) filters/chips still shown, matching
    # the same "not yet applied" convention as confirm/clarify.
    assert resp.filters == []

    # The real, expected path: recruiter checks a subset and hits Search,
    # submitting [domain_filter, skill in [checked]] via the existing
    # deterministic PATCH endpoint -- no LLM call for the actual search.
    from app.core.skill_taxonomy import skill_names_of
    from app.models.schemas import PatchStateRequest
    apply_resp = svc.patch_state(PatchStateRequest(
        job_id=JOB_DEVOPS, session_id="s1",
        filters=[resp.domain_filter, Filter(field="skill", operator="in", value=["Docker"])],
        logic="AND",
    ))
    assert apply_resp.status == "ok"
    # domain is now credit-eligible for partial-match the same way skill
    # already was (see _fuzzy_skill_matches) -- a candidate hitting ONE of
    # {domain="DevOps", skill=Docker} but not the other is shown as a
    # partial match, honestly labeled with whichever one is missing, rather
    # than silently excluded. Full matches (no partial_skill_match tag)
    # must still ALL have Docker literally; partial matches may be missing
    # exactly Docker (matched via DevOps domain instead) or exactly "DevOps
    # experience" (matched via Docker instead).
    full = [c for c in apply_resp.candidates if not c.get("partial_skill_match")]
    assert full and all("Docker" in skill_names_of(c) for c in full)
    for c in apply_resp.candidates:
        pm = c.get("partial_skill_match")
        if pm:
            assert pm["missing"] in (["Docker"], ["DevOps experience"])


def test_domain_only_query_skips_skill_pick_when_no_real_overlap():
    # A domain term the taxonomy DOES recognize (real tools exist) but
    # where none of them appear among this job's real domain-matching
    # candidates must fall through to a normal search, not offer an empty
    # or meaningless choice list. Confirmed on real data: "fintech" has
    # ~100 real taxonomy tools (Stripe, PayPal, ...), but zero overlap with
    # JOB_DEVOPS's real fintech-classified candidates' actual skills.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="domain", operator="contains", value="fintech")])
    svc = make_service(out)
    resp = svc.filter_by_query("who has worked in fintech", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.status != "domain_skill_pick"


def test_domain_search_shows_domain_specific_years_not_total_career():
    # Real, reported UX gap: "who has worked in fintech" showed the
    # candidate's TOTAL career length on their card, not years actually
    # spent in that domain. Real data on JOB_DEVOPS: Adam N Schmidt has
    # 12.4 years of total career experience but only 0.8 years classified
    # as "Payments & FinTech Engineering" -- the card must surface the
    # 0.8, not silently imply he has 12.4 years of fintech experience.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="domain", operator="contains", value="fintech")])
    svc = make_service(out)
    resp = svc.filter_by_query("who has worked in fintech", job_id=JOB_DEVOPS, session_id="s1")
    schmidt = next(c for c in resp.candidates if c["name"] == "Adam N Schmidt")
    assert schmidt["experience"] == 12.4
    assert schmidt["domain_match_years"] == [
        {"domain": "Payments & FinTech Engineering", "years": 0.8}
    ]

    # A query with NO domain/domain_experience filter never gets this tag,
    # even for the same candidate.
    out2 = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals", value=schmidt["location"])])
    svc2 = make_service(out2)
    resp2 = svc2.filter_by_query("location test", job_id=JOB_DEVOPS, session_id="s2")
    schmidt2 = next(c for c in resp2.candidates if c["name"] == "Adam N Schmidt")
    assert "domain_match_years" not in schmidt2


def test_abandoned_sessions_are_swept_not_retained_forever():
    # Expiry used to be purely lazy -- an entry was only dropped when that
    # SAME key was looked up again -- so the common case (one search, then
    # the browser is closed, key never touched again) retained the session
    # for the life of the process. Each entry holds a whole SessionState
    # including `last_candidates` (every candidate dict that was on screen),
    # so abandoned sessions accumulated real memory indefinitely.
    store = InMemorySessionStore(ttl=-1)  # everything is immediately expired
    store.set("abandoned", "job1", SessionState())
    assert "abandoned::job1" in store._data

    store._last_sweep = 0.0  # make the throttled sweep due on the next write
    store.set("active", "job1", SessionState())
    assert "abandoned::job1" not in store._data  # swept without ever being read
    assert "active::job1" in store._data  # written after the sweep, so retained


def test_or_word_between_requires_both_values_present_and_or_strictly_between():
    q = "Find candidates who know Kubernetes or Terraform"
    assert service_module._or_word_between(q, "Kubernetes", "Terraform") is True
    assert service_module._or_word_between(q, "Terraform", "Kubernetes") is True  # order-independent


def test_or_word_between_false_when_or_is_unrelated_to_the_pair():
    # The "or" here belongs to Mumbai/Delhi, not Python/Django -- must not
    # be read as applying to an unrelated pair elsewhere in the query.
    q = "Python and Django experience, based in Mumbai or Delhi"
    assert service_module._or_word_between(q, "Python", "Django") is False


def test_or_word_between_false_when_a_value_is_not_found_verbatim():
    # "K8s" in the query, not "Kubernetes" -- can't verify the trigger, so
    # it must stay quiet rather than guess (same principle as
    # taxonomy._mentioned_in_query).
    q = "knows K8s or Terraform"
    assert service_module._or_word_between(q, "Kubernetes", "Terraform") is False


def test_collapse_same_field_or_pairs_dedupes_exact_duplicates_unconditionally():
    # Real, reported live bug: "Kubernetes or Terraform" alone produced
    # FOUR filters, the same two values each listed twice. A literal
    # duplicate must always collapse, "or" present or not.
    filters = [
        Filter(field="skill", operator="contains", value="Kubernetes", hard=True),
        Filter(field="skill", operator="contains", value="Terraform", hard=True),
        Filter(field="skill", operator="contains", value="Kubernetes", hard=True),
        Filter(field="skill", operator="contains", value="Terraform", hard=True),
    ]
    out, note = service_module._collapse_same_field_or_pairs(filters, "knows Kubernetes and Terraform")
    assert len(out) == 2
    assert {f.operator for f in out} == {"contains"}  # no "or" in query -> not collapsed to "in"
    assert note is None


def test_collapse_same_field_or_pairs_never_dedupes_two_different_skills(monkeypatch=None):
    # Real, SELF-CAUGHT bug found while verifying the fix above live: two
    # DIFFERENT skill_experience filters (Kubernetes, Terraform) that both
    # happen to carry the same `value=0` (an incomplete/unspecified-years
    # marker -- see _repair_incomplete_skill_experience) looked like exact
    # duplicates of EACH OTHER once the dedup key didn't also check
    # `f.skill` -- silently discarding Terraform's requirement entirely,
    # keeping only Kubernetes. `f.skill` (not `f.value`) is the real
    # identity for skill_experience/domain_experience filters, exactly as
    # Filter.key() already establishes.
    filters = [
        Filter(field="skill_experience", operator="gte", value=0, skill="Kubernetes", hard=True),
        Filter(field="skill_experience", operator="gte", value=0, skill="Terraform", hard=True),
    ]
    out, note = service_module._collapse_same_field_or_pairs(
        filters, "Kubernetes or Terraform experience",
    )
    assert {f.skill for f in out} == {"Kubernetes", "Terraform"}
    assert note is None  # skill_experience isn't contains/equals -- not grouped/collapsed here


def test_collapse_same_field_or_pairs_collapses_when_query_says_or():
    # The actual originally-reported live bug, exact shape: "Kubernetes or
    # Terraform" produced two same-field `contains` filters, each hard=True
    # -- duplicated, so four total -- which apply_spec would AND together
    # (must have BOTH), the opposite of what "or" asked for.
    filters = [
        Filter(field="skill", operator="contains", value="Kubernetes", hard=True),
        Filter(field="skill", operator="contains", value="Terraform", hard=True),
        Filter(field="skill", operator="contains", value="Kubernetes", hard=True),
        Filter(field="skill", operator="contains", value="Terraform", hard=True),
    ]
    out, note = service_module._collapse_same_field_or_pairs(
        filters, "Find candidates who know Kubernetes or Terraform",
    )
    assert len(out) == 1
    assert out[0].operator == "in"
    assert set(out[0].value) == {"Kubernetes", "Terraform"}
    assert "either" in note


def test_collapse_same_field_or_pairs_leaves_a_genuine_and_alone():
    # Same filter SHAPE as the bug above (two same-field `contains`
    # filters) but the OPPOSITE intended meaning -- "needs both Python and
    # Django" is a real, common, genuinely-AND request. Only the query's
    # own wording (no "or" here) can tell the two apart.
    filters = [
        Filter(field="skill", operator="contains", value="Python", hard=True),
        Filter(field="skill", operator="contains", value="Django", hard=True),
    ]
    out, note = service_module._collapse_same_field_or_pairs(
        filters, "need someone who knows both Python and Django",
    )
    assert out == filters
    assert note is None


def test_collapse_same_field_or_pairs_ignores_an_unrelated_or_elsewhere():
    filters = [
        Filter(field="skill", operator="contains", value="Python", hard=True),
        Filter(field="skill", operator="contains", value="Django", hard=True),
    ]
    out, note = service_module._collapse_same_field_or_pairs(
        filters, "Python and Django experience, based in Mumbai or Delhi",
    )
    assert out == filters
    assert note is None


def test_kubernetes_terraform_duplicate_bug_collapses_to_or_end_to_end():
    # End-to-end regression for the actual live bug (see
    # _collapse_same_field_or_pairs's docstring): reproduces the exact
    # malformed LLMOutput observed live and confirms the final response
    # exposes a single OR filter, not two independently-mandatory ones.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND", filters=[
        Filter(field="skill", operator="contains", value="Kubernetes", hard=True),
        Filter(field="skill", operator="contains", value="Terraform", hard=True),
        Filter(field="skill", operator="contains", value="Kubernetes", hard=True),
        Filter(field="skill", operator="contains", value="Terraform", hard=True),
    ])
    svc = make_service(out)
    resp = svc.filter_by_query(
        "Find candidates who know Kubernetes or Terraform", job_id=JOB, session_id="s1",
    )
    assert len(resp.filters) == 1
    assert resp.filters[0].operator == "in"
    assert set(resp.filters[0].value) == {"Kubernetes", "Terraform"}


def test_bare_skill_experience_mention_is_never_trusted_without_a_real_number(monkeypatch=None):
    # Real, reported live bug: a bare "<skill> experience" mention with NO
    # number stated anywhere ("Candidates with Python experience") was
    # unreliably routed to `skill_experience(Python, gte, 1)` -- the model
    # invented "1" to satisfy the shape it picked, not because any number
    # was said. Reproduces the exact malformed structured item observed
    # live; the query itself has zero digits/number-words anywhere, so the
    # threshold is fabricated by construction and must be dropped to a
    # plain skill filter, not treated as "at least 1 year".
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND", filters=[
        Filter(field="skill_experience", operator="gte", value=1, skill="Python", hard=True),
    ])
    svc = make_service(out)
    resp = svc.filter_by_query("Candidates with Python experience", job_id=JOB, session_id="s1")
    assert resp.filters == [Filter(field="skill", operator="contains", value="Python", hard=True)]


def test_genuine_skill_experience_number_is_never_second_guessed():
    # The other half of the same fix, tested directly against the function
    # (not the full pipeline, which has its own SEPARATE, unrelated
    # "no real per-skill years DATA exists" fallback further downstream --
    # see _skill_years_available -- that would otherwise muddy what this
    # specific fix is responsible for): a REAL stated threshold must
    # survive repair untouched, since the query here genuinely contains a
    # number and is NOT the fabricated-value case above.
    filters = [Filter(field="skill_experience", operator="gte", value=2, skill="Python", hard=True)]
    out, _ = service_module._repair_resolved_filters(
        filters, "Candidates with at least 2 years of Python experience",
    )
    assert out == filters


def test_results_rank_exact_before_fuzzy_full_before_partial():
    # Real, reported bug: a fuzzy/related-tool match (satisfies EVERY
    # requirement, just not literally -- see fuzzy_skill_match) could
    # outrank a genuine 100%-exact match purely because its own, unrelated
    # match_score happened to be higher -- confirmed live on job_id=00000103
    # for "Project Managers: Agile + Jira, 10+ years": Anthony Carthen Cell
    # (Jira satisfied only via a related tool) sorted FIRST, ahead of 6 real
    # exact matches, before this fix. Tiers must never interleave regardless
    # of match_score.
    #
    # UPDATED counts: job_title is now credit-eligible for partial-match the
    # same way skill already was (real, explicit request -- a candidate
    # hitting every skill but missing the stated role no longer vanishes
    # outright). Real data: 6 exact (job_title, experience, AND both skills
    # literally), 1 fuzzy-full (Anthony Carthen Cell -- Jira via a related
    # tool, title still literal), 32 partial (missing exactly one of job_title/
    # Agile/Jira -- previously only 7, since a job_title mismatch used to be
    # an outright exclusion rather than partial credit).
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[
                        Filter(field="job_title", operator="contains", value="Project Manager"),
                        Filter(field="skill", operator="contains", value="Agile"),
                        Filter(field="skill", operator="contains", value="Jira"),
                        Filter(field="experience", operator="gte", value=10),
                    ])
    svc = make_service(out)
    resp = svc.filter_by_query(
        "Project Managers: Agile + Jira, 10+ years", job_id=JOB_DEVOPS, session_id="s1",
    )
    assert resp.showing == 39
    # partial_skill_match takes precedence in classification: a partial-tier
    # candidate can ALSO carry a fuzzy_skill_match tag (matched one skill via
    # a related tool while missing another requirement entirely) without
    # that promoting them out of the partial tier.
    tiers = [
        "partial" if c.get("partial_skill_match") else "fuzzy" if c.get("fuzzy_skill_match") else "exact"
        for c in resp.candidates
    ]
    assert tiers == (
        ["exact"] * 6 + ["fuzzy"] * 1 + ["partial"] * 32
    )


def test_fuzzy_full_match_is_tagged_not_indistinguishable_from_exact():
    # Real data on JOB_DEVOPS: querying "kubernetes" surfaces 12 candidates,
    # 6 of whom qualify only via a curated related-tool relation (see
    # skill_taxonomy.related_terms_for), not the literal word "kubernetes"
    # in their skill list. Before this test's original fix, a candidate
    # promoted via _fuzzy_skill_matches's `full_extra` path (unlike
    # `partial`) was merged with ZERO distinction from a real exact match --
    # confirmed live: a job_id=00000103 search for "3 years of exp in java"
    # surfaced a candidate with NO Java anywhere in their resume, silently
    # passed off as an ordinary result. `fuzzy_skill_match` closes that gap
    # for the deterministic (taxonomy-related) case; the LLM-verified
    # semantic case this test used to also cover was later removed entirely
    # (see service.py's comment on _fuzzy_skill_matches) since it proved
    # non-functional on this project's model.
    #
    # (This test originally used "python" on JOB -- since fixed to exclude
    # cross-LANGUAGE relations from counting as "related" at all, see
    # skill_taxonomy._PROGRAMMING_LANGUAGES: real, reported live bug, a
    # mechanical/controls engineer whose only point of contact with "Python"
    # was knowing MATLAB -- a different language entirely, not a Python
    # library/framework -- got surfaced as a fuzzy "Python" match. That fix
    # correctly zeroed out JOB's old fuzzy set entirely (it was ALL
    # language-pair false positives), so this test now uses "kubernetes" on
    # JOB_DEVOPS instead to keep exercising a genuine, still-valid
    # tool-to-tool relation.)
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="kubernetes")])
    svc = make_service(out)
    resp = svc.filter_by_query("kubernetes", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.showing == 12

    tagged = {c["name"]: c["fuzzy_skill_match"] for c in resp.candidates if c.get("fuzzy_skill_match")}
    assert tagged == {
        name: [{"skill": "Kubernetes", "matched_via": "related"}]
        for name in (
            "Arvin Aloumian", "Andrew Jones", "Adam Hardesty",
            "Anthony Carthen Cell", "Arun Kumar Reddy", "Aaron Landis",
        )
    }
    # Exact matches (the literal word "kubernetes" in their own skill list)
    # carry no tag at all -- only the fuzzy ones do.
    exact_names = {c["name"] for c in resp.candidates} - set(tagged)
    assert "Arthur Zaslawski" in exact_names
    for name in exact_names:
        matching = next(c for c in resp.candidates if c["name"] == name)
        assert "fuzzy_skill_match" not in matching or not matching["fuzzy_skill_match"]


def test_same_field_skill_in_filter_gets_fuzzy_widening_too():
    # Regression: "Kubernetes or Terraform"-style same-field alternatives
    # resolve to `skill in [...]` (see prompt.py rule 4), not
    # alternative_groups -- but that `in` filter used to be checked by
    # apply_spec/_op_in via pure literal string equality only, bypassing
    # _fuzzy_skill_matches entirely (its skill_idx only looked at
    # "contains"/"not_contains"). Same real data as the "kubernetes"
    # contains test above (12 total: 6 exact + 6 related-tool-only) -- an
    # `in` filter naming "kubernetes" must now surface the identical 12, not
    # just the 6 literal matches. (A second, unrecognized value in the list
    # would instead trip skill_taxonomy's separate "genuine umbrella
    # concept" expansion path -- out of scope here, which targets only the
    # operator="in" fuzzy-widening gap itself.)
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="in", value=["kubernetes"])])
    svc = make_service(out)
    resp = svc.filter_by_query("kubernetes", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.showing == 12
    fuzzy_names = {c["name"] for c in resp.candidates if c.get("fuzzy_skill_match")}
    assert fuzzy_names == {
        "Arvin Aloumian", "Andrew Jones", "Adam Hardesty",
        "Anthony Carthen Cell", "Arun Kumar Reddy", "Aaron Landis",
    }


def test_alternative_group_skill_branch_gets_fuzzy_widening_too():
    # Regression: a skill filter living inside an alternative_groups branch
    # (e.g. "either a PhD or hands-on Kubernetes experience") used to be
    # matched by apply_spec's group-gate literally only -- see
    # _fuzzy_alternative_group_matches's docstring. Same real "kubernetes"
    # data (12 total on JOB_DEVOPS) via a single-branch group instead of a
    # flat filter.
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND", filters=[],
                    alternative_groups=[AlternativeGroup(filters=[
                        Filter(field="skill", operator="contains", value="kubernetes"),
                    ])])
    svc = make_service(out)
    resp = svc.filter_by_query("kubernetes route", job_id=JOB_DEVOPS, session_id="s1")
    assert resp.showing == 12
    fuzzy_names = {c["name"] for c in resp.candidates if c.get("fuzzy_skill_match")}
    assert fuzzy_names == {
        "Arvin Aloumian", "Andrew Jones", "Adam Hardesty",
        "Anthony Carthen Cell", "Arun Kumar Reddy", "Aaron Landis",
    }


def test_fuzzy_skill_match_label_dedupes_when_value_list_has_alias_duplicates():
    # Real, reported live bug: a filter's own value list can contain
    # several different-looking entries that all canonicalize() to the SAME
    # name -- confirmed live case: v2's "expand" match_mode
    # (taxonomy._tool_to_filter) calls expand_skill_term() directly, with
    # NO dedup pass afterward (unlike the v1 repair path's
    # expand_skill_filters, which already collapses alias duplicates via
    # is_known_tool + _dedupe before a filter ever reaches this function --
    # confirmed by hand: routing this exact value list through
    # filter_by_query's normal v1 pipeline masks the bug entirely, so this
    # test calls _fuzzy_skill_matches directly to reproduce the real,
    # un-deduped v2 shape). Before expand_skill_term's own alias-bloat fix,
    # that path could hand this function a value list like ["machine
    # learning", "ML model", "ML pipeline", ..., "Scikit Learn", ...] where
    # 18 entries all canonicalize back to "machine learning" -- the
    # un-deduped join rendered a fuzzy_skill_match badge as "machine
    # learning/machine learning/...(18x).../Scikit Learn/...". Reproduces
    # that shape with Kubernetes' own real aliases (confirmed via
    # canonicalize: "Kubernetes"/"K8s"/"kube" all resolve to "Kubernetes" --
    # this test originally used Python's aliases on JOB, but that pool's
    # only fuzzy matches for Python were cross-language false positives
    # since excluded, see skill_taxonomy._PROGRAMMING_LANGUAGES).
    svc = make_service(LLMOutput(intent="FILTER_CANDIDATES"))
    spec = FilterSpec(logic="AND", filters=[
        Filter(field="skill", operator="in",
               value=["Kubernetes", "K8s", "kube"]),
    ])
    full_extra, _partial = svc._fuzzy_skill_matches(JOB_DEVOPS, spec, matched_ids=set())
    tagged = [c for c in full_extra if c.get("fuzzy_skill_match")]
    assert tagged  # real data: 6 candidates qualify only via a related tool
    for c in tagged:
        assert c["fuzzy_skill_match"] == [{"skill": "Kubernetes", "matched_via": "related"}]


def test_session_state_persists_and_merges():
    store = InMemorySessionStore()
    # First: has Python (8 of the 111 real candidates for this job, all
    # literal matches -- see skill_taxonomy._PROGRAMMING_LANGUAGES for why
    # this pool's Python search no longer surfaces any fuzzy related-tool
    # matches: they were all cross-language false positives, e.g. a
    # candidate who only knew MATLAB or C++, now correctly excluded).
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains",
                                    value="python")])),
        store=store,
    )
    r1 = svc1.filter_by_query("python", job_id=JOB, session_id="s1")
    assert r1.showing == 8

    # Then: add location Mumbai. This ADDS a genuinely new field on top of
    # an already-active search -- a pure addition drops nothing existing,
    # so it auto-applies immediately with no confirm step (see the
    # dropped_keys-gated confirm condition in _filter_by_query: a real,
    # reported live complaint was that EVERY change to the active filter
    # set stopped the recruiter with a checklist, even a completely
    # unambiguous "also X" that couldn't possibly lose anything).
    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals",
                                    value="Mumbai")])),
        store=store,
    )
    r2 = svc2.filter_by_query("mumbai", job_id=JOB, session_id="s1")
    assert r2.status == "ok"
    assert {f.key() for f in r2.filters} == {
        Filter(field="skill", operator="contains", value="Python").key(),
        Filter(field="location", operator="equals", value="Mumbai").key(),
    }
    # AND with existing Python -> 1 (Abhijeet B Kshirsagar, exact).
    assert r2.showing == 1


def test_confirm_does_not_fire_on_a_same_field_value_update():
    # "Actually Mumbai instead of Bangalore" -- the field SET is unchanged
    # (still just {location}), only its value differs. Nothing about WHICH
    # filters are active is in question, so this auto-applies with no
    # confirm step -- the scope decision this feature was built around.
    # Real match counts for either city are irrelevant to what's being
    # tested here (only that "confirm" never fires on a same-key update).
    store = InMemorySessionStore()
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals", value="Bangalore")])),
        store=store,
    )
    r1 = svc1.filter_by_query("bangalore", job_id=JOB, session_id="s1")
    assert r1.status != "confirm"

    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals", value="Mumbai")])),
        store=store,
    )
    r2 = svc2.filter_by_query("actually mumbai instead", job_id=JOB, session_id="s1")
    assert r2.status != "confirm"
    assert r2.filters == [Filter(field="location", operator="equals", value="Mumbai")]


def test_confirm_fires_on_a_dropped_field_with_default_unchecked():
    # replace_all=True and the new output genuinely omits the old field --
    # it still gets a row (never silently vanishes), but defaults
    # unchecked since replace_all said drop it.
    store = InMemorySessionStore()
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="python")])),
        store=store,
    )
    svc1.filter_by_query("python", job_id=JOB, session_id="s1")

    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND", replace_all=True,
                    filters=[Filter(field="location", operator="equals", value="Mumbai")])),
        store=store,
    )
    r2 = svc2.filter_by_query("mumbai", job_id=JOB, session_id="s1")
    assert r2.status == "confirm"
    by_field = {c.filter.field: c for c in r2.choices}
    assert by_field["skill"].origin == "existing" and by_field["skill"].default_checked is False
    assert by_field["location"].origin == "new" and by_field["location"].default_checked is True


def test_confirm_response_carries_extra_message():
    # extra_message (LLM's own note, or taxonomy skip_notes) must survive
    # into the confirm response's message, not be silently dropped. Uses a
    # replace_all drop (see test_confirm_fires_on_a_dropped_field_with_
    # default_unchecked's docstring for why that's the only case that still
    # triggers confirm at all -- a pure addition, like the "also Mumbai" this
    # test originally used, now auto-applies immediately with no confirm
    # step, since nothing is at risk of being silently lost there).
    store = InMemorySessionStore()
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="python")])),
        store=store,
    )
    svc1.filter_by_query("python", job_id=JOB, session_id="s1")

    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND", replace_all=True,
                    filters=[Filter(field="location", operator="equals", value="Mumbai")],
                    message="Salary filtering isn't supported.")),
        store=store,
    )
    r2 = svc2.filter_by_query("actually just mumbai, high salary", job_id=JOB, session_id="s1")
    assert r2.status == "confirm"
    assert "Salary filtering isn't supported." in r2.message


def test_confirm_choices_submitted_via_patch_state_is_the_real_ui_path():
    # The actual expected interaction: recruiter unchecks a box in the UI
    # and hits Search, which calls PATCH /ai/candidates/filter/state
    # directly -- no LLM, no "yes"/"no" text at all. Confirms the plan's
    # claim that patch_state needs ZERO changes to serve as the apply step.
    # Uses a replace_all drop to reach confirm at all -- see
    # test_confirm_response_carries_extra_message's docstring.
    from app.models.schemas import PatchStateRequest

    store = InMemorySessionStore()
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="python")])),
        store=store,
    )
    svc1.filter_by_query("python", job_id=JOB, session_id="s1")

    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND", replace_all=True,
                    filters=[Filter(field="location", operator="equals", value="Mumbai")])),
        store=store,
    )
    r2 = svc2.filter_by_query("actually just mumbai", job_id=JOB, session_id="s1")
    assert r2.status == "confirm"

    # Recruiter re-checks "skill" (keeping both) and hits Search.
    kept = [c.filter for c in r2.choices]
    r3 = svc2.patch_state(PatchStateRequest(
        job_id=JOB, session_id="s1", filters=kept, logic=r2.logic,
    ))
    assert r3.status == "ok"
    assert {f.key() for f in r3.filters} == {
        Filter(field="location", operator="equals", value="Mumbai").key(),
        Filter(field="skill", operator="contains", value="Python").key(),
    }


def test_confirm_bare_no_asks_what_instead_and_clears_pending_state():
    # Uses a replace_all drop to reach confirm at all -- see
    # test_confirm_response_carries_extra_message's docstring.
    store = InMemorySessionStore()
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="skill", operator="contains", value="python")])),
        store=store,
    )
    svc1.filter_by_query("python", job_id=JOB, session_id="s1")
    svc2 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND", replace_all=True,
                    filters=[Filter(field="location", operator="equals", value="Mumbai")])),
        store=store,
    )
    svc2.filter_by_query("actually just mumbai", job_id=JOB, session_id="s1")

    svc3 = FilterService(llm=svc2.llm, store=store)
    r3 = svc3.filter_by_query("no", job_id=JOB, session_id="s1")
    assert r3.status == "clarify"
    assert r3.question == "Okay -- what would you like instead?"
    # "no" reverts to whatever was already active BEFORE the just-proposed
    # merge -- the proposed Mumbai-only replacement was only ever a pending
    # suggestion, never persisted, so only the original skill filter remains.
    assert r3.filters == [Filter(field="skill", operator="contains", value="Python")]


def test_lookup_answers_from_real_data_after_narrowing_to_one():
    store = InMemorySessionStore()
    # Narrow to exactly one real candidate. Mumbai + python alone now
    # matches 2 (Abhijeet B Kshirsagar, exact; Ashwani Kumar, a related-tool
    # match added by the info.json taxonomy migration -- see
    # skill_taxonomy.py's module docstring), so a third real filter
    # (Ashwani Kumar has 0 years experience) re-narrows to exactly one,
    # confirmed unique in this job's pool.
    svc1 = FilterService(
        llm=FakeLLM(LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[Filter(field="location", operator="equals", value="Mumbai"),
                             Filter(field="skill", operator="contains", value="python"),
                             Filter(field="experience", operator="gte", value=1)])),
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
    assert r1.showing == 8
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
    monkeypatch.setattr(service_module.experience_index, "index_exists", lambda *a, **k: False)

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
    monkeypatch.setattr(service_module.experience_index, "index_exists", lambda *a, **k: True)

    def fake_search(query, candidate_ids, top_k=20):
        # Pool-scoping is search_candidates' OWN contract now (restrict
        # before ranking -- see experience_index.py), verified directly in
        # tests/test_experience_search.py; this fake simulates that
        # contract (only returning ids service.py actually passed in) so
        # this test can verify the OTHER half -- that service.py passes
        # the real job pool through and doesn't itself let anything
        # unscoped slip past the fake.
        candidates = {
            _REAL_ID_1: 0.81,
            _FAKE_ID: 0.95,  # not in JOB -- a real search_candidates would never return this
        }
        return [{"candidate_id": cid, "score": s}
                for cid, s in candidates.items() if cid in candidate_ids]
    monkeypatch.setattr(service_module.experience_index, "search_candidates", fake_search)

    out = LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="led a team of engineers")
    svc = make_service(out)
    resp = svc.filter_by_query("who led a team of engineers", job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    ids = {c["id"] for c in resp.candidates}
    assert ids == {_REAL_ID_1}
    assert resp.candidates[0]["experience_match_score"] == 0.81


def test_experience_search_keeps_best_score_per_candidate(monkeypatch):
    monkeypatch.setattr(service_module.experience_index, "index_exists", lambda *a, **k: True)

    def fake_search(query, candidate_ids, top_k=20):
        # Same candidate, two matching experience chunks -- best score wins.
        return [
            {"candidate_id": _REAL_ID_1, "score": 0.60},
            {"candidate_id": _REAL_ID_1, "score": 0.88},
        ]
    monkeypatch.setattr(service_module.experience_index, "search_candidates", fake_search)

    out = LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="built a payment system")
    svc = make_service(out)
    resp = svc.filter_by_query("who built a payment system", job_id=JOB, session_id="s1")
    assert resp.showing == 1
    assert resp.candidates[0]["experience_match_score"] == 0.88


def test_experience_search_no_hits_returns_no_match(monkeypatch):
    monkeypatch.setattr(service_module.experience_index, "index_exists", lambda *a, **k: True)
    monkeypatch.setattr(service_module.experience_index, "search_candidates", lambda query, candidate_ids, top_k=20: [])

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
    monkeypatch.setattr(service_module.experience_index, "index_exists", lambda *a, **k: True)

    floor = service_module.FilterService._EXPERIENCE_MIN_SIMILARITY
    below_floor = floor - 0.01
    above_floor = floor + 0.10

    def fake_search(query, candidate_ids, top_k=20):
        return [
            {"candidate_id": _REAL_ID_1, "score": above_floor},
            {"candidate_id": _REAL_ID_2, "score": below_floor},
        ]
    monkeypatch.setattr(service_module.experience_index, "search_candidates", fake_search)

    out = LLMOutput(intent="EXPERIENCE_SEARCH", experience_query="led a team")
    svc = make_service(out)
    resp = svc.filter_by_query("who led a team", job_id=JOB, session_id="s1")
    ids = {c["id"] for c in resp.candidates}
    assert ids == {_REAL_ID_1}


def test_experience_search_intersects_with_active_structured_filter(monkeypatch):
    monkeypatch.setattr(service_module.experience_index, "index_exists", lambda *a, **k: True)

    def fake_search(query, candidate_ids, top_k=20):
        # Both real candidates match semantically...
        return [
            {"candidate_id": _REAL_ID_1, "score": 0.90},
            {"candidate_id": _REAL_ID_2, "score": 0.85},
        ]
    monkeypatch.setattr(service_module.experience_index, "search_candidates", fake_search)

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
    """Real, reported live bug: "backend engineer who worked on X" used to
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
    monkeypatch.setattr(service_module.experience_index, "index_exists", lambda *a, **k: True)

    def fake_search(query, candidate_ids, top_k=20):
        # Both real candidates match semantically...
        return [
            {"candidate_id": _REAL_ID_1, "score": 0.90},  # Manohar Patil
            {"candidate_id": _REAL_ID_2, "score": 0.85},  # Ganesh B Shelke
        ]
    monkeypatch.setattr(service_module.experience_index, "search_candidates", fake_search)

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
# schema_v2 migration: hard/soft preferences (Filter.hard) and the v2
# extraction path (structured/tools -> app/core/taxonomy.py -> Filter).
# --------------------------------------------------------------------------- #
from app.core.merge import to_chips
from app.core.service import _hard_only
from app.models.schemas import FilterSpec, StructuredItem, ToolItem


def test_hard_only_drops_soft_filters_under_and_logic():
    spec = FilterSpec(logic="AND", filters=[
        Filter(field="location", operator="equals", value="Mumbai", hard=True),
        Filter(field="skill", operator="contains", value="Docker", hard=False),
    ])
    narrowed = _hard_only(spec)
    assert [f.field for f in narrowed.filters] == ["location"]


def test_hard_only_is_a_noop_with_no_soft_filters():
    spec = FilterSpec(logic="AND", filters=[
        Filter(field="location", operator="equals", value="Mumbai", hard=True),
    ])
    assert _hard_only(spec) is spec  # no copy made when there's nothing to drop


def test_hard_only_ignores_the_hard_flag_under_or_logic():
    # A soft filter inside an OR set is a full alternative branch of
    # inclusion, not an optional extra -- pulling it out would change what
    # the query MEANS, not just how results are ordered. See _hard_only's
    # docstring.
    spec = FilterSpec(logic="OR", filters=[
        Filter(field="location", operator="equals", value="Mumbai", hard=True),
        Filter(field="skill", operator="contains", value="Docker", hard=False),
    ])
    narrowed = _hard_only(spec)
    assert len(narrowed.filters) == 2  # both kept, hard flag ignored under OR


def test_hard_only_ignores_the_hard_flag_under_not_logic():
    spec = FilterSpec(logic="NOT", filters=[
        Filter(field="skill", operator="contains", value="Docker", hard=False),
    ])
    assert len(_hard_only(spec).filters) == 1


def test_apply_soft_preferences_reranks_without_excluding():
    candidates = [
        {"id": "no-docker", "skills": ["python"], "match_score": 90},
        {"id": "has-docker", "skills": ["python", "docker"], "match_score": 50},
    ]
    soft = [Filter(field="skill", operator="contains", value="Docker", hard=False)]
    result = FilterService._apply_soft_preferences(candidates, soft)
    # Nobody excluded -- soft preferences only re-rank survivors.
    assert {c["id"] for c in result} == {"no-docker", "has-docker"}
    # The one satisfying the soft preference ranks first, despite a lower
    # match_score -- soft-match count is the primary sort key here.
    assert result[0]["id"] == "has-docker"
    assert result[0]["soft_match"] == {
        "satisfied": [c.label for c in to_chips(soft)], "missing": [], "count": 1,
    }
    assert result[1]["soft_match"]["count"] == 0
    assert result[1]["soft_match"]["missing"]


def test_apply_soft_preferences_ties_preserve_original_order():
    # Neither candidate satisfies the soft preference -- order must be
    # unchanged (stable sort), not reshuffled.
    candidates = [{"id": "a", "skills": [], "match_score": 10},
                 {"id": "b", "skills": [], "match_score": 20}]
    soft = [Filter(field="skill", operator="contains", value="Docker", hard=False)]
    result = FilterService._apply_soft_preferences(candidates, soft)
    assert [c["id"] for c in result] == ["a", "b"]


def test_soft_skill_preference_reranks_real_candidates_without_excluding():
    # End-to-end through filter_by_query: a hard location filter plus a
    # soft skill preference. Real data, real engine, no LLM call needed
    # (FakeLLM scripts the output).
    out = LLMOutput(intent="FILTER_CANDIDATES", logic="AND",
                    filters=[
                        Filter(field="skill", operator="contains", value="python", hard=True),
                        Filter(field="skill", operator="contains", value="docker", hard=False),
                    ])
    svc = make_service(out)
    resp = svc.filter_by_query("python devs, docker preferred", job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    # Every returned candidate has python (the hard filter still excludes);
    # docker is a preference, so its presence/absence must not have
    # narrowed the result -- just reordered it. soft_match must be present.
    assert all("soft_match" in c for c in resp.candidates)


class FakeLLMv2:
    """Same scripted-output contract as FakeLLM, but with prompt_schema set
    so service.py routes FILTER_CANDIDATES through app/core/taxonomy.py
    instead of the v1 expand_skill_filters path -- see
    service.py's `getattr(self.llm, "prompt_schema", "v1")` check."""
    prompt_schema = "v2"

    def __init__(self, output: LLMOutput):
        self.output = output
        self.calls = 0

    def translate(self, query, current_filters, history=None):
        self.calls += 1
        return self.output


def test_v2_tools_and_structured_resolve_through_taxonomy_to_real_matches():
    out = LLMOutput(intent="FILTER_CANDIDATES", replace_all=True,
                    tools=[ToolItem(raw_text="Python", match_mode="exact", hard=True)])
    svc = FilterService(llm=FakeLLMv2(out), store=InMemorySessionStore())
    resp = svc.filter_by_query("candidates who know Python", job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    assert resp.showing > 0
    assert resp.filters[0].field == "skill"
    assert resp.filters[0].operator == "contains"


def test_v2_structured_item_resolves_to_a_real_hard_filter():
    out = LLMOutput(intent="FILTER_CANDIDATES", replace_all=True,
                    structured=[StructuredItem(field="location", operator="equals",
                                               raw_text="Mumbai", value="Mumbai", hard=True)])
    svc = FilterService(llm=FakeLLMv2(out), store=InMemorySessionStore())
    resp = svc.filter_by_query("candidates in Mumbai", job_id=JOB, session_id="s1")
    assert resp.status == "ok"
    assert resp.filters == [Filter(field="location", operator="equals", value="Mumbai", hard=True)]


def test_v2_domain_hint_is_logged_not_filtered():
    # domain_hint must not silently become a Filter (see taxonomy.py's
    # scoping note) -- it should surface as a message, and must not affect
    # who matches.
    out = LLMOutput(intent="FILTER_CANDIDATES", replace_all=True,
                    structured=[StructuredItem(field="location", operator="equals",
                                               raw_text="Mumbai", value="Mumbai", hard=True)],
                    domain_hint=["Finance"])
    svc = FilterService(llm=FakeLLMv2(out), store=InMemorySessionStore())
    resp = svc.filter_by_query("finance folks in Mumbai", job_id=JOB, session_id="s1")
    assert len(resp.filters) == 1  # domain_hint did not become a second filter
    assert resp.message and "Finance" in resp.message


def test_v2_concept_wrapper_word_and_typo_resolve_identically_through_the_real_dispatch():
    # End-to-end proof (not just a skill_taxonomy.py unit test) that a
    # noise-wrapped mention ("machine learning concepts") and a typo
    # ("muchine learning") reach a real recruiter through the ACTUAL v2
    # service dispatch (FakeLLMv2, not a bare FakeLLM -- see
    # service.py's `getattr(self.llm, "prompt_schema", "v1")` check, which
    # a plain FakeLLM silently fails and routes through the v1 path
    # instead, masking whether taxonomy.resolve_filters's fixes actually
    # get invoked at all). Real, reported live bug: "machine learning
    # concepts" resolved to a literal, unrecognized filter that matched
    # ZERO candidates on JOB_DEVOPS, where the bare "machine learning"
    # concept correctly matches 11 real candidates.
    def run(raw_text: str):
        out = LLMOutput(intent="FILTER_CANDIDATES", replace_all=True,
                        tools=[ToolItem(raw_text=raw_text, match_mode="expand", hard=True)])
        svc = FilterService(llm=FakeLLMv2(out), store=InMemorySessionStore())
        return svc.filter_by_query(f"knows {raw_text}", job_id=JOB_DEVOPS, session_id=raw_text)

    baseline = run("machine learning")
    assert baseline.status == "ok"
    assert baseline.showing == 11

    for variant in ("machine learning concepts", "muchine learning"):
        resp = run(variant)
        assert resp.status == "ok"
        assert resp.showing == 11
        assert {c["name"] for c in resp.candidates} == {c["name"] for c in baseline.candidates}


def test_a_narrowing_phrase_overrides_a_spurious_replace_all():
    # Real, reported live bug: turn 1 sets a Python filter; turn 2, "out of
    # all give me guy from high tire comapany", is unambiguously a
    # NARROWING of the already-active results ("out of all [these
    # candidates]...") -- but the real model set replace_all=true anyway,
    # which would silently offer to DROP the active Python filter (either
    # via a confirm screen that could be missed, or -- since a pure
    # addition no longer needs confirming at all, see
    # test_session_state_persists_and_merges -- silently, immediately)
    # instead of combining it with the new company-tier filter, exactly the
    # opposite of what the recruiter's own words asked for. The override
    # makes this a genuine merge (nothing dropped), so it now auto-applies
    # BOTH filters immediately with no confirm step at all -- an even
    # better outcome than a confirm screen the recruiter has to notice and
    # act on.
    store = InMemorySessionStore()
    turn1 = LLMOutput(intent="FILTER_CANDIDATES", replace_all=True,
                      tools=[ToolItem(raw_text="Python", match_mode="exact", hard=True)])
    svc = FilterService(llm=FakeLLMv2(turn1), store=store)
    r1 = svc.filter_by_query("candidates who know python", job_id=JOB_DEVOPS, session_id="s-narrow")
    assert r1.status == "ok"

    turn2 = LLMOutput(intent="FILTER_CANDIDATES", replace_all=True,
                      structured=[StructuredItem(field="company_tier", operator="gte",
                                                 value="High", hard=True)])
    svc.llm = FakeLLMv2(turn2)
    r2 = svc.filter_by_query(
        "out of all give me guy from high tire comapany", job_id=JOB_DEVOPS, session_id="s-narrow",
    )
    assert r2.status == "ok"
    assert {f.key() for f in r2.filters} == {
        Filter(field="skill", operator="contains", value="Python", hard=True).key(),
        Filter(field="company_tier", operator="gte", value="High", hard=True).key(),
    }


def test_narrowing_phrase_override_is_a_no_op_with_nothing_active():
    # A fresh session (spec.filters empty) has nothing to preserve, so the
    # override must not change behavior -- replace_all=true still resolves
    # to a normal "ok" search, not an accidental confirm/merge artifact.
    out = LLMOutput(intent="FILTER_CANDIDATES", replace_all=True,
                    tools=[ToolItem(raw_text="Python", match_mode="exact", hard=True)])
    svc = FilterService(llm=FakeLLMv2(out), store=InMemorySessionStore())
    resp = svc.filter_by_query(
        "out of all candidates who know python", job_id=JOB_DEVOPS, session_id="s-narrow-fresh",
    )
    assert resp.status == "ok"
    assert resp.filters == [Filter(field="skill", operator="contains", value="Python", hard=True)]
