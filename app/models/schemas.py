"""Pydantic models: API contracts and internal filter representation."""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Filter primitives
# --------------------------------------------------------------------------- #
# Fields where multiple distinct values must coexist as separate filters
# rather than the newest silently replacing the oldest -- "knows React and
# Python" needs BOTH kept, not whichever the LLM listed last. Fields not
# listed here (location, education, experience, ...) keep single-value
# "replace" semantics -- "actually, Bangalore instead" should replace, not
# stack, a location. job_title/certification are multi-value for the same
# reason as university/company -- a candidate can hold several of each, and
# "worked as Manager and Team Lead" / "has AWS and Scrum certs" need both.
_MULTI_VALUE_FIELDS = {"skill", "university", "company", "job_title", "certification"}


class Filter(BaseModel):
    field: str
    operator: str
    value: Union[str, int, float, bool, list[Any]]
    skill: Optional[str] = None
    unit: Optional[str] = None
    # True (default) = a hard requirement, excludes non-matching candidates,
    # exactly today's behavior for every existing caller (PATCH endpoint,
    # PendingCombine-stored specs, every pre-v2 construction site) -- the
    # default is what makes this field additive rather than a breaking
    # change. False = a stated preference (schema_v2's "nice to have"): does
    # NOT exclude anyone, only re-ranks survivors of the hard filters upward
    # when they also satisfy it -- see service.py's _apply_soft_preferences.
    # Only meaningful under AND logic (see _hard_only's docstring for why
    # OR/NOT ignore this flag and treat every filter as hard).
    hard: bool = True

    def key(self) -> tuple:
        """Identity for merge/dedup: a location replaces a location, but two
        different skills (or universities, or companies) coexist.

        `skill` (this attribute) only carries a value for `skill_experience`
        filters (which skill the *years* refer to) -- a plain `field="skill"`
        filter puts the skill name in `value` instead, so distinguishing by
        `value` is what actually keeps "React" and "Python" as two filters
        instead of colliding on the same key.
        """
        if self.field in _MULTI_VALUE_FIELDS:
            return (self.field, str(self.value).lower())
        return (self.field, (self.skill or "").lower())


class AlternativeGroup(BaseModel):
    """One eligibility "route" in an OR-of-AND-groups requirement, e.g.
    "Master's from a Tier-1 university OR 10+ years total experience" is two
    AlternativeGroups: [education gte Master, college_tier gte High] and
    [experience gte 10]. A candidate passes the group requirement if they
    satisfy ALL filters in AT LEAST ONE group (groups are OR'd against each
    other; each group's own filters are AND'd). A group cannot contain
    another group -- verified against every real "or" pattern in this
    project's few-shots and the triggering query that motivated this model:
    none needs deeper nesting, and this also sidesteps any question of
    whether Ollama's grammar compiler supports self-referential JSON schema,
    by construction.

    Exists because top-level `FilterSpec.logic` is ONE flat operator over
    the ENTIRE `filters` list (see engine.apply_spec) -- setting it to "OR"
    to express one embedded alternative would wrongly turn every OTHER
    AND'd requirement in the same query into an optional alternative too.
    Same-field alternatives ("AWS or Azure") don't need this at all -- the
    existing "in" operator already covers those; this is only for a genuine
    cross-field "either requirement route A or route B" embedded alongside
    other hard requirements.

    Every filter here is forced hard=True at validation time (see
    validation.validate_alternative_groups) -- a soft preference has no
    meaning inside an eligibility route; a route is satisfied or not.

    (A parallel, independently-built implementation of this same "either
    requirement route A or B" idea existed briefly on another branch shaped
    as ONE group holding `branches: list[list[Filter]]` instead of one
    group per route -- semantically equivalent, reconciled to this shape
    since the rest of this codebase's alternative_groups handling --
    merge_alternative_groups's wholesale-replace semantics, the seniority-
    band feature, every existing test -- was already built around it.)"""
    filters: list[Filter] = Field(default_factory=list)


class FilterSpec(BaseModel):
    logic: Literal["AND", "OR", "NOT"] = "AND"
    filters: list[Filter] = Field(default_factory=list)
    # Additional gate, ANDed on top of `filters`/`logic` above -- see
    # AlternativeGroup's docstring. Only meaningful combined with
    # logic=="AND", same restriction/rationale as Filter.hard (see
    # service._hard_only's docstring) -- under top-level OR/NOT there's no
    # coherent way to compose "OR of top-level filters" with "AND-gate on
    # top of that", so this is dropped whenever logic != "AND".
    alternative_groups: list[AlternativeGroup] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# schema_v2 extraction shapes (PROMPT_SCHEMA=v2 only)
#
# The v2 prompt never asks the model to resolve a raw span into a final
# filter value -- it only reports WHAT WAS SAID (a span, a coarse bucket,
# hard-vs-soft, exact-vs-expand). app/core/taxonomy.py::resolve_filters()
# turns these into real `Filter` objects, reusing skill_taxonomy.py's
# canonicalize/expand_skill_term exactly as the v1 path already does --
# these two model classes are the intermediate, unresolved representation,
# never handed to engine.py/validation.py directly.
# --------------------------------------------------------------------------- #
class StructuredItem(BaseModel):
    field: str
    operator: str
    raw_text: Optional[str] = None
    value: Union[str, int, float, bool, list[Any]]
    hard: bool = True
    # ONLY populated when field == "skill_experience" -- which named skill
    # the number of years refers to (mirrors Filter.skill's same role).
    skill: Optional[str] = None


class ToolItem(BaseModel):
    raw_text: str
    # "exact": one specific named tool was demanded -- resolved via
    # skill_taxonomy.canonicalize() only (alias fix, never widened).
    # "expand": the recruiter signaled flexibility ("familiar with X",
    # "like Y") -- resolved via skill_taxonomy.expand_skill_term(), which
    # widens to related tools when the taxonomy has something to say, and
    # falls back to canonicalize()-only when it doesn't.
    match_mode: Literal["exact", "expand"] = "exact"
    hard: bool = True


# --------------------------------------------------------------------------- #
# LLM output (before validation / merge)
# --------------------------------------------------------------------------- #
class LLMOutput(BaseModel):
    intent: str
    logic: Literal["AND", "OR", "NOT"] = "AND"
    filters: list[Filter] = Field(default_factory=list)
    # v2-only (empty under v1): raw extraction buckets, resolved into
    # `filters`-equivalent Filter objects by app/core/taxonomy.py, never
    # consumed directly by validation.py/engine.py. `domain_hint` is
    # deliberately NOT turned into a Filter yet -- the real candidate
    # `domain` field stores 212 fine-grained subdomain strings ("FinTech",
    # "Payments & FinTech Engineering"), while this is a coarse 14-category
    # hint ("Finance") that often shares no substring with the real data at
    # all (engine.py matches `domain` by substring) -- captured/logged for
    # now rather than shipped as a silently-inert soft filter. See the
    # migration plan's domain_hint scoping note for the follow-up.
    structured: list[StructuredItem] = Field(default_factory=list)
    tools: list[ToolItem] = Field(default_factory=list)
    domain_hint: list[str] = Field(default_factory=list)
    # A genuine cross-field "either requirement route A or route B" embedded
    # alongside other AND'd requirements (see AlternativeGroup's docstring)
    # -- e.g. "8+ years AND [Master's from a Tier-1 university OR 10+ years
    # of experience]". Members carry ALREADY-RESOLVED field/operator/value
    # (v1's shape) even under v2 -- see json_schema_v2.py's comment on why
    # group leaves deliberately bypass the raw-span structured/tools layer.
    alternative_groups: list[AlternativeGroup] = Field(default_factory=list)
    # Set true ONLY when NEW QUERY reads as a full standalone search that
    # doesn't build on CURRENT FILTERS at all (e.g. CURRENT FILTERS has
    # location+experience+skill and NEW QUERY is just "candidates in
    # mumbai", mentioning none of the others) -- tells the backend to
    # REPLACE the whole filter set with `filters` instead of merging field-
    # by-field. False (default) for anything that reads as refining/adding
    # to what's already active ("also add Python", "actually, Bangalore
    # instead", "and 5+ years too") -- those still merge normally. See
    # prompt.py rule 1b.
    replace_all: bool = False
    # CLARIFY -- a genuinely ambiguous query gets a follow-up question
    # instead of a guess. clarify_field/clarify_operator (when the question
    # is about a concrete threshold on one ALLOWED_FIELDS field, e.g.
    # "experience") let the backend resolve the recruiter's next reply
    # ("2+ years", or clicking that exact option) DETERMINISTICALLY -- by
    # extracting the number, not by re-sending the bare reply to the LLM
    # with no memory of what was asked, which doesn't reliably work (a
    # fragment like "2+ years" alone often isn't enough for the model to
    # know what field it answers, especially the poorer the model). Left
    # None for clarifications that don't reduce to one field+threshold
    # (e.g. "show me good candidates" -- which criterion isn't decided yet).
    question: Optional[str] = None
    options: list[str] = Field(default_factory=list)
    clarify_field: Optional[str] = None
    clarify_skill: Optional[str] = None
    clarify_operator: Optional[str] = None
    # Set ONLY for a CONFIRM-style clarify -- one where a specific candidate
    # value is already known and the question is just asking the recruiter
    # to confirm/deny it (e.g. "Should the experience be at least 7 years?"
    # -> clarify_value=7), as opposed to an OPEN clarify with no candidate
    # value yet (e.g. "How many years of experience?"). Lets a bare "yes"/
    # "no" reply resolve deterministically in code (see PendingClarify.value
    # and service.py) instead of needing the LLM to re-derive a number that
    # was only ever stated in the natural-language question text -- which it
    # structurally cannot recover from "yes" alone with no memory of it.
    clarify_value: Optional[Union[str, int, float]] = None
    clarify_unit: Optional[str] = None
    # UNSUPPORTED_FILTER
    message: Optional[str] = None
    # LOOKUP -- a question about ONE specific already-shown candidate, not a
    # new filter. candidate_ref: whatever text identifies who ("he", "the
    # first one", a name). lookup_field: which ALLOWED_FIELDS field they're
    # asking about (e.g. "university" for "which college did he go to").
    # The backend resolves both deterministically against the real,
    # already-fetched candidate data -- the LLM never states the answer
    # itself, only which question is being asked.
    candidate_ref: Optional[str] = None
    lookup_field: Optional[str] = None
    # EXPERIENCE_SEARCH -- the query asks whether candidates DID something
    # specific in their work (a project, responsibility, achievement) that
    # isn't a named skill/tool/title/certification, e.g. "led a team of
    # engineers", "built a payment processing system". This has no
    # SINGLE structured field to translate into on its own -- it's matched
    # against the actual sentences of each candidate's real job history via
    # semantic search (see app/core/experience_index.py), not a filter.
    # experience_query is the phrase to search for, in the recruiter's own
    # words -- pass it through close to verbatim, don't try to normalize it
    # into a keyword.
    #
    # A compound ask CAN also carry ordinary `filters` in the SAME turn when
    # the query separately names a real structured requirement alongside the
    # achievement -- e.g. "backend engineers who worked on a supply chain
    # platform" names BOTH a job_title AND an achievement. Put the
    # structured part in `filters` exactly as FILTER_CANDIDATES would, and
    # the achievement phrase in `experience_query`, still under intent
    # EXPERIENCE_SEARCH (the achievement is the harder, more specific part
    # that decides the intent) -- the backend applies both: `filters`
    # narrows the pool first, `experience_query` semantically searches
    # within it (see service.py's _answer_experience_search). Confirmed
    # live this used to be silently dropped entirely: "backend engineer who
    # worked on X" only ever searched X, matching non-engineers too. Only
    # emit `filters` here for a genuine separate requirement named in the
    # SAME sentence -- don't invent one, same rule 6a-i discipline as
    # everywhere else.
    experience_query: Optional[str] = None


# --------------------------------------------------------------------------- #
# API request / response
# --------------------------------------------------------------------------- #
class FilterRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    job_id: str
    session_id: str
    reset: bool = False  # clear session state before applying this query


class PatchStateRequest(BaseModel):
    """Deterministic chip removal / direct filter edit — no LLM involved."""
    job_id: str
    session_id: str
    filters: list[Filter]
    logic: Literal["AND", "OR", "NOT"] = "AND"
    alternative_groups: list[AlternativeGroup] = Field(default_factory=list)


class Chip(BaseModel):
    label: str
    field: str
    skill: Optional[str] = None
    # Mirrors the Filter it was rendered from (see Filter.hard) -- a real
    # flag the UI can key off of (dashed border, different color, etc.),
    # not just a text convention baked into the label string.
    hard: bool = True


class FilterChoice(BaseModel):
    """One row of a "confirm" response -- a single filter the recruiter can
    keep or drop, shown individually rather than folded into one bundled
    yes/no sentence. `label` is `merge.chip_label(filter)` verbatim (never
    hand-rolled) so the same "~" soft-preference convention and any future
    hard/soft styling comes for free. `origin` distinguishes a filter
    carried over from the active session ("existing") from one the new
    query just introduced ("new") -- purely informational for the UI, not
    used to decide `default_checked` (see FilterService's confirm-branch
    docstring for why hard/soft doesn't change the default either)."""
    filter: Filter
    label: str
    origin: Literal["existing", "new"]
    default_checked: bool


class FilterResponse(BaseModel):
    status: Literal[
        "ok", "clarify", "confirm", "domain_skill_pick",
        "unsupported", "no_match", "error", "answer",
    ]
    total: int = 0
    showing: int = 0
    logic: str = "AND"
    filters: list[Filter] = Field(default_factory=list)
    alternative_groups: list[AlternativeGroup] = Field(default_factory=list)
    chips: list[Chip] = Field(default_factory=list)
    candidates: list[dict] = Field(default_factory=list)
    # clarify
    question: Optional[str] = None
    options: list[str] = Field(default_factory=list)
    # confirm -- see FilterChoice and PendingConfirm
    choices: list[FilterChoice] = Field(default_factory=list)
    # domain_skill_pick -- see PendingDomainSkillPick. `domain_filter` is the
    # domain/domain_experience filter the recruiter's query already named
    # (not yet applied -- `filters`/`chips` above still reflect the OLD,
    # pre-this-turn state, same convention as confirm/clarify); the
    # recruiter checks zero or more of `skill_options` and submits the
    # final [existing chips + domain_filter + optional skill filter] via
    # the existing deterministic PATCH endpoint.
    domain_filter: Optional[Filter] = None
    skill_options: list[str] = Field(default_factory=list)
    # unsupported / error / no_match / answer
    message: Optional[str] = None
    suggestions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Session state: active filters AND who was last shown, so a follow-up like
# "which college did he go to" can be resolved to a real candidate record.
# --------------------------------------------------------------------------- #
class PendingClarify(BaseModel):
    """What a CLARIFY question was actually about, so the next reply
    ("2+ years", or clicking that exact option) can be turned into a real
    filter deterministically -- extracting the number, never re-sending the
    bare reply to the LLM with no memory of the question."""
    field: str
    operator: str
    skill: Optional[str] = None
    # Set for a CONFIRM-style clarify (see LLMOutput.clarify_value) -- lets
    # a bare "yes"/"no" reply resolve deterministically: yes -> apply this
    # exact value, no LLM call, no chance of hallucinating a value that was
    # only ever in the question's natural-language text.
    value: Optional[Union[str, int, float]] = None
    unit: Optional[str] = None


class PendingConfirm(BaseModel):
    """Every filter the recruiter is being asked to individually keep or
    drop, awaiting their reply, before a query is actually applied.

    Set whenever a new query would ADD or DROP a field on top of an
    already-active, non-empty filter set (e.g. active filter is "Mumbai",
    new query asks for "high tier college" -- a different field entirely)
    -- as opposed to updating a field already present ("actually, Bangalore
    instead"), which still auto-applies without asking, since nothing about
    WHICH filters are active is in question there.

    Superseded PendingCombine, which asked ONE bundled yes/no question
    spanning every filter in the merged set ("Do you want candidates
    matching fintech domain and python skill?"). Confirmed live that this
    hides a real bug: a stale filter from an earlier, unrelated query
    survived because it was bundled into a sentence the recruiter agreed to
    without noticing it was in there. Granularity is the actual fix -- each
    filter is its own row (see FilterChoice), not smarter guessing about
    when to ask.

    The recruiter's real answer is a set of checkboxes (each `FilterChoice`
    carries `default_checked` as the pre-ticked suggestion), submitted via
    the existing PATCH /ai/candidates/filter/state endpoint -- no LLM call
    needed for that path at all. This pending state exists ONLY so a bare
    "yes"/"no" typed in the chat box (rather than clicking checkboxes) still
    resolves deterministically: "yes" applies every `default_checked`
    filter verbatim, same principle as PendingClarify.value.

    `alternative_groups` (see AlternativeGroup) carries this turn's new
    group statement, if any, ALONGSIDE the flat-filter review above --
    reviewing/dropping individual routes is explicitly out of scope for
    now (a stated "either route" always auto-applies, same as any other
    unreviewed field), this field exists only so a group statement isn't
    silently LOST when a flat-filter change in the same turn also happens
    to trigger this confirm step."""
    choices: list[FilterChoice]
    logic: Literal["AND", "OR", "NOT"] = "AND"
    message: Optional[str] = None
    alternative_groups: list[AlternativeGroup] = Field(default_factory=list)


class PendingDomainSkillPick(BaseModel):
    """A vague domain-only query ("I want a DevOps guy") named a practice
    area but no specific tool -- rather than searching immediately (which
    would just mean "anyone ever classified into this domain", the same
    complaint that led to domain_experience existing at all), offer real,
    data-grounded skill choices to narrow it first (see service.py's
    _domain_skill_options: the curated tool taxonomy's list for that
    practice area, intersected with skills that ACTUALLY appear among this
    job's real candidates classified into it -- never a tool nobody in this
    pool actually has).

    Exists so a bare short reply typed in the chat box can still resolve
    this deterministically, same principle as PendingClarify/PendingConfirm
    -- the real, expected path is the recruiter checking boxes in the UI
    and hitting Search, which submits straight to the existing
    deterministic PATCH endpoint (no LLM call needed for that path)."""
    domain_filter: Filter
    skill_options: list[str]
    message: Optional[str] = None


class ChatTurn(BaseModel):
    """One turn of real conversation history, sent back to the LLM verbatim
    as prior chat messages (not summarized/hand-parsed) -- so a short reply
    like "yes"/"no"/"actually make it 6" resolves against whatever was
    actually just said, generally, instead of needing a hand-coded
    extractor for every possible clarify shape. The deterministic
    fast-paths (pending_lookup_field, pending_clarify) still short-circuit
    the common cases without an LLM call; history is what makes the LLM
    fallback actually capable for everything else, instead of failing."""
    role: Literal["user", "assistant"]
    content: str


class SessionState(BaseModel):
    spec: FilterSpec = Field(default_factory=FilterSpec)
    last_candidates: list[dict] = Field(default_factory=list)
    # Set when a LOOKUP was ambiguous (multiple candidates could match) and
    # we asked "which one?". If the very next message names one of them, it
    # completes THIS lookup directly -- bypassing the LLM entirely, since a
    # bare name has no other sensible interpretation as a fresh query, and
    # the LLM has no way to recover "which field were we even asking about"
    # from a bare name alone.
    pending_lookup_field: Optional[str] = None
    # Same idea for CLARIFY: set whenever the LLM identified which field a
    # clarifying question was about (see LLMOutput.clarify_field).
    pending_clarify: Optional[PendingClarify] = None
    # Set when a query adds or drops a field on top of an already-active
    # search, awaiting the recruiter's per-filter keep/drop confirmation
    # -- see PendingConfirm's docstring.
    pending_confirm: Optional[PendingConfirm] = None
    # Set when a query named a domain/practice-area with no specific skill,
    # awaiting the recruiter's optional skill narrowing -- see
    # PendingDomainSkillPick's docstring.
    pending_domain_skill_pick: Optional[PendingDomainSkillPick] = None
    # Recent real conversation turns (bounded, see service._append_history),
    # replayed to the LLM as actual prior chat messages on every call.
    history: list[ChatTurn] = Field(default_factory=list)
