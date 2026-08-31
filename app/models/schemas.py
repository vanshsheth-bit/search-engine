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
    # CLARIFY
    question: Optional[str] = None
    options: list[str] = Field(default_factory=list)
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
