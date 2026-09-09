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
    """A bounded, non-recursive way to express "either this WHOLE set of
    requirements, or that one" spanning DIFFERENT fields -- e.g. "a Master's
    from a Tier-1 university OR 10+ years of experience". Deliberately NOT a
    fully general nested boolean tree: real recruiter queries essentially
    never need more than "everything else required, AND at least one of
    these alternative requirement-sets also holds" -- one extra level, not
    arbitrary depth. A single field's own list of alternative VALUES
    ("Mumbai, Pune, or Bangalore") does NOT need this at all -- that's a
    single Filter with operator "in" (the engine already evaluates "in"
    correctly against any field, list-valued or scalar, ordinal or not; see
    engine.matches_filter's final fallback). AlternativeGroup exists only
    for the genuinely cross-field case a single "in" list cannot express.

    Every branch is itself an AND of one or more filters (e.g. the
    "Master's from Tier-1" branch is TWO filters -- education AND
    college_tier -- that must both hold for that branch to count). The
    group as a whole is satisfied if ANY branch's filters ALL match."""
    branches: list[list[Filter]] = Field(default_factory=list)


class FilterSpec(BaseModel):
    logic: Literal["AND", "OR", "NOT"] = "AND"
    filters: list[Filter] = Field(default_factory=list)
    # ANDed against `filters` above -- every group here must be satisfied
    # (by at least one of ITS branches) in addition to everything in
    # `filters`. See AlternativeGroup's docstring.
    alternative_groups: list[AlternativeGroup] = Field(default_factory=list)
    # See LLMOutput.preferred_filters -- never excludes anyone, just ranked/
    # noted. Persisted separately so it survives across turns the same way
    # `filters` does.
    preferred_filters: list[Filter] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# LLM output (before validation / merge)
# --------------------------------------------------------------------------- #
class LLMOutput(BaseModel):
    intent: str
    logic: Literal["AND", "OR", "NOT"] = "AND"
    filters: list[Filter] = Field(default_factory=list)
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
    # See AlternativeGroup's docstring -- an "either this WHOLE requirement-
    # set or that one" ask spanning DIFFERENT fields (e.g. "a Master's from
    # a Tier-1 university OR 10+ years of experience"). ANDed against
    # `filters` above. A single field's own list of alternative VALUES
    # ("Mumbai, Pune, or Bangalore") does NOT belong here -- that's a
    # single ordinary Filter in `filters` with operator "in" instead.
    alternative_groups: list[AlternativeGroup] = Field(default_factory=list)
    # Soft-preference language ("prefer", "ideally", "bonus if", "nice to
    # have") is NOT the same as a requirement, and must never silently
    # become one -- confirmed live: "prefer candidates ... in Mumbai, Pune,
    # or Bangalore" got folded into the same hard AND as everything else,
    # zeroing out real candidates who matched every actual requirement but
    # happened to be elsewhere. Put PREFERRED (not required) criteria here
    # instead of in `filters` -- the backend never excludes anyone for
    # failing one of these, it only notes honestly that the preference
    # couldn't be strictly enforced (see service.py). Same Filter shape as
    # `filters`, just a different bucket with different (non-exclusionary)
    # semantics.
    preferred_filters: list[Filter] = Field(default_factory=list)


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


class Chip(BaseModel):
    label: str
    field: str
    skill: Optional[str] = None
    # True for a chip built from an AlternativeGroup ("Master's (Tier-1)
    # OR 10+ yrs exp") or from `preferred_filters` -- the UI renders these
    # visually distinct from an ordinary hard-required chip, since neither
    # kind excludes a candidate the way a normal filter does.
    preferred: bool = False
    alternative: bool = False


class FilterResponse(BaseModel):
    status: Literal["ok", "clarify", "unsupported", "no_match", "error", "answer"]
    total: int = 0
    showing: int = 0
    logic: str = "AND"
    filters: list[Filter] = Field(default_factory=list)
    alternative_groups: list[AlternativeGroup] = Field(default_factory=list)
    preferred_filters: list[Filter] = Field(default_factory=list)
    chips: list[Chip] = Field(default_factory=list)
    candidates: list[dict] = Field(default_factory=list)
    # clarify
    question: Optional[str] = None
    options: list[str] = Field(default_factory=list)
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


class PendingCombine(BaseModel):
    """A proposed filter combination awaiting yes/no confirmation before
    being applied. Set when a new query introduces at least one genuinely
    NEW field on top of an already-active, non-empty filter set (e.g.
    active filter is "Mumbai", new query asks for "high tier college" --
    a different field entirely) -- as opposed to updating a field already
    present ("actually, Bangalore instead"), which still auto-replaces
    without asking, or a query that reads as a full standalone search
    (see LLMOutput.replace_all), which still replaces the whole set
    without asking. Only the "silently stack an unrelated new requirement
    onto the existing search" case is ambiguous enough to warrant a
    confirmation -- the recruiter might have meant EITHER "Mumbai AND
    high tier college" OR "actually, forget Mumbai, just high tier
    college" and guessing either way risks a wrong result the recruiter
    has no reason to expect.

    Holds the FULLY MERGED spec, ready to apply verbatim on "yes" -- no
    LLM call needed to resolve the confirmation, same principle as
    PendingClarify.value."""
    spec: FilterSpec
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
    # Set when a query introduces a genuinely new field on top of an
    # already-active search, awaiting yes/no confirmation before combining
    # -- see PendingCombine's docstring.
    pending_combine: Optional[PendingCombine] = None
    # Recent real conversation turns (bounded, see service._append_history),
    # replayed to the LLM as actual prior chat messages on every call.
    history: list[ChatTurn] = Field(default_factory=list)
