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


class FilterSpec(BaseModel):
    logic: Literal["AND", "OR", "NOT"] = "AND"
    filters: list[Filter] = Field(default_factory=list)


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


class FilterResponse(BaseModel):
    status: Literal["ok", "clarify", "unsupported", "no_match", "error", "answer"]
    total: int = 0
    showing: int = 0
    logic: str = "AND"
    filters: list[Filter] = Field(default_factory=list)
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
    # Recent real conversation turns (bounded, see service._append_history),
    # replayed to the LLM as actual prior chat messages on every call.
    history: list[ChatTurn] = Field(default_factory=list)
