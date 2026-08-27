"""Deterministic resolution for the LOOKUP intent: which candidate, which
fact. The LLM only identifies WHAT is being asked (candidate_ref,
lookup_field) -- it never states the answer itself. This module resolves
candidate_ref against the real last-shown candidates and formats the real
stored value, reusing engine.extract_value so field lookup here can never
drift from how the exact same field is read during filtering.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.engine import extract_value
from app.core.vocabulary import FIELD_LABELS
from app.models.schemas import Filter


@dataclass
class LookupResolution:
    candidate: dict | None = None
    ambiguous_names: list[str] | None = None  # set when >1 candidate could match


def _name_matches(ref_words: set[str], name: str) -> bool:
    """Every word in the reference must appear as a whole word in the
    candidate's name -- so "deep mehta" matches "Deep Paresh Mehta" (a
    plain substring check would miss this: "deep mehta" is not a
    contiguous substring when a middle name sits between them)."""
    name_words = set(name.lower().split())
    return bool(ref_words) and ref_words.issubset(name_words)


def resolve_candidate(ref: str | None, last_candidates: list[dict]) -> LookupResolution:
    """Resolve a reference like "he" / "the first one" / a name against the
    candidates actually shown last turn. Only a name match is attempted --
    ordinal phrasing ("the first one") isn't resolved here; ambiguous cases
    fall through to asking rather than guessing."""
    if not last_candidates:
        return LookupResolution()
    if len(last_candidates) == 1:
        return LookupResolution(candidate=last_candidates[0])

    ref_words = set((ref or "").strip().lower().split())
    if ref_words:
        matches = [c for c in last_candidates if _name_matches(ref_words, c.get("name") or "")]
        if len(matches) == 1:
            return LookupResolution(candidate=matches[0])
        if len(matches) > 1:
            return LookupResolution(ambiguous_names=[c.get("name", "Unnamed") for c in matches])

    return LookupResolution(ambiguous_names=[c.get("name", "Unnamed") for c in last_candidates])


def _format_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "none listed"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def answer_lookup(candidate: dict, field: str) -> str:
    """Build the answer sentence entirely from the candidate's real stored
    data -- never from the LLM's own words."""
    name = candidate.get("name") or "This candidate"
    if field not in FIELD_LABELS:
        return f"I don't track \"{field}\" for candidates."
    label = FIELD_LABELS[field]
    value = extract_value(candidate, Filter(field=field, operator="equals", value=""))
    if value in (None, [], ""):
        return f"I don't have {label} on file for {name}."
    return f"{name}'s {label}: {_format_value(value)}."
