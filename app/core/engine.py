"""Deterministic filter engine.

Pure functions. The LLM never runs here. Match scores are read-only: filtering
selects a subset and preserves original ordering/score.

Candidate shape expected (flexible — missing fields are treated as absent):
{
    "id": "c1",
    "name": "Asha",
    "match_score": 92,
    "location": "Mumbai",
    "experience": 6,                       # total years
    "education": "Master",
    "university": ["KJ Somaiya School of Engineering, Mumbai"],
    "college_tier": "Low",         # Low / Medium / High
    "company": ["Deutsche Bank", "Wipro"],
    "company_tier": "High",        # Low / Medium / High, best of their companies
    "notice_period_days": 15,
    "relocation": true,
    "skills": {"Python": {"years": 5}, "React": {"years": 2}}
    # skills may also be a flat list: ["Python", "React"]
}
"""
from __future__ import annotations

import logging

from app.core.vocabulary import college_tier_rank, company_tier_rank, education_rank
from app.models.schemas import Filter, FilterSpec

logger = logging.getLogger(__name__)

# Fields compared by canonical rank, not literal string identity -- so
# "Masters"/"Master's"/"MS" all match the same candidates regardless of
# phrasing, and "High"/"HIGH"/"high" tier all resolve the same way.
_ORDINAL_RANK_FIELDS = {
    "education": education_rank,
    "college_tier": college_tier_rank,
    "company_tier": company_tier_rank,
}

# List fields holding free-text names (institution/company names, not atomic
# tokens) -- "contains" must be a substring match here ("Somaiya" must match
# "KJ Somaiya School of Engineering, Mumbai"), unlike `skill`, where each
# list entry is already an atomic token and a substring match would wrongly
# let "java" match "javascript".
_FREE_TEXT_LIST_FIELDS = {"university", "company"}


# --------------------------------------------------------------------------- #
# Operator implementations
# --------------------------------------------------------------------------- #
def _num(x) -> float:
    return float(x)


def _op_equals(cand, val) -> bool:
    return str(cand).strip().lower() == str(val).strip().lower()


def _op_contains(cand, val, substring_for_list: bool = False) -> bool:
    if isinstance(cand, (list, tuple, set)):
        if substring_for_list:
            # Free-text list entries (university names): "Somaiya" must match
            # "KJ Somaiya School of Engineering, Mumbai" as a substring.
            return any(str(val).lower() in str(c).lower() for c in cand)
        # Atomic-token list entries (skills): exact match per item, so "java"
        # does NOT wrongly match "javascript" as a substring.
        return any(str(val).lower() == str(c).lower() for c in cand)
    return str(val).lower() in str(cand).lower()


OPERATORS = {
    "equals": _op_equals,
    "not_equals": lambda c, v: not _op_equals(c, v),
    "contains": _op_contains,
    "not_contains": lambda c, v: not _op_contains(c, v),
    "gte": lambda c, v: _num(c) >= _num(v),
    "lte": lambda c, v: _num(c) <= _num(v),
    "gt": lambda c, v: _num(c) > _num(v),
    "lt": lambda c, v: _num(c) < _num(v),
    "in": lambda c, v: any(_op_equals(c, x) for x in v),
    "not_in": lambda c, v: not any(_op_equals(c, x) for x in v),
}


# --------------------------------------------------------------------------- #
# Field value extraction
# --------------------------------------------------------------------------- #
def _skills_map(candidate: dict) -> dict[str, dict]:
    """Normalise skills into {name: {'years': n}} regardless of input shape."""
    raw = candidate.get("skills")
    if isinstance(raw, dict):
        out = {}
        for name, meta in raw.items():
            if isinstance(meta, dict):
                out[name.lower()] = meta
            else:  # {"Python": 5}
                out[name.lower()] = {"years": meta}
        return out
    if isinstance(raw, (list, tuple)):
        return {str(s).lower(): {"years": None} for s in raw}
    return {}


def _notice_days(candidate: dict):
    for key in ("notice_period_days", "notice_period", "noticePeriodDays"):
        if key in candidate and candidate[key] is not None:
            return candidate[key]
    return None


def extract_value(candidate: dict, f: Filter):
    """Return the candidate's value for the filter's field, or None if absent."""
    field = f.field

    if field == "skill":
        return list(_skills_map(candidate).keys())

    if field == "skill_experience":
        skill = (f.skill or "").lower()
        meta = _skills_map(candidate).get(skill)
        if meta is None:
            return None
        return meta.get("years")

    if field == "notice_period":
        return _notice_days(candidate)

    if field == "education":
        edu = candidate.get("education")
        return edu

    return candidate.get(field)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def matches_filter(candidate: dict, f: Filter) -> bool:
    value = extract_value(candidate, f)

    # Absent data:
    #   - skill / skill_experience: candidate simply doesn't have it -> fail
    #     positive checks; "not_contains" on a missing skill should pass.
    if value is None:
        if f.operator in {"not_contains", "not_equals", "not_in"}:
            return True
        return False

    op = OPERATORS.get(f.operator)
    if op is None:
        logger.warning("Unknown operator %s", f.operator)
        return False

    try:
        rank_fn = _ORDINAL_RANK_FIELDS.get(f.field)
        if rank_fn is not None:
            cand_rank = rank_fn(value)
            want_rank = rank_fn(f.value) if not isinstance(f.value, list) else None
            if f.operator in {"gte", "lte", "gt", "lt"}:
                if cand_rank is None or want_rank is None:
                    return False
                return op(cand_rank, want_rank)
            if f.operator == "equals":
                return cand_rank is not None and cand_rank == want_rank
            if f.operator == "not_equals":
                return cand_rank is None or want_rank is None or cand_rank != want_rank
            if f.operator in {"in", "not_in"}:
                want_ranks = {rank_fn(v) for v in f.value}
                hit = cand_rank is not None and cand_rank in want_ranks
                return hit if f.operator == "in" else not hit
            return False
        if f.operator in {"contains", "not_contains"}:
            hit = _op_contains(value, f.value, substring_for_list=f.field in _FREE_TEXT_LIST_FIELDS)
            return hit if f.operator == "contains" else not hit
        return op(value, f.value)
    except (ValueError, TypeError):
        return False


def apply_spec(candidates: list[dict], spec: FilterSpec) -> list[dict]:
    """Apply the full filter spec with AND/OR/NOT logic. Scores untouched."""
    if not spec.filters:
        return list(candidates)

    kept = []
    for c in candidates:
        checks = [matches_filter(c, f) for f in spec.filters]
        if spec.logic == "OR":
            keep = any(checks)
        elif spec.logic == "NOT":
            keep = not any(checks)
        else:  # AND
            keep = all(checks)
        if keep:
            kept.append(c)

    # Preserve match-score ordering (highest first). Original scores unchanged.
    return sorted(kept, key=lambda c: c.get("match_score", 0), reverse=True)
