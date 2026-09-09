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
# let "java" match "javascript". job_title/certification are the same shape
# as university/company: free-text strings, not atomic tokens (a certification
# entry can be a whole sentence naming several certs at once). `domain` is
# the same shape too -- a candidate's real values are specific subdomain
# names ("FinTech", "Payments & FinTech Engineering"), and "fintech" must
# match either as a substring, not just the exact shorter one.
_FREE_TEXT_LIST_FIELDS = {"university", "company", "job_title", "certification", "domain"}


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


def _op_in(cand, values, substring_for_list: bool = False) -> bool:
    """"in" against a LIST-valued candidate field (skill/university/company/...)
    means "does the candidate have ANY of these values", checked per-item the
    same way "contains" is -- NOT a naive str(whole_list) == str(one_value)
    comparison, which would compare a stringified list against a single term
    and (correctly) almost never match. Needed for skill-concept expansion:
    a candidate matches "in" a list like ["machine learning", "tensorflow",
    "pytorch"] if their skills contain ANY one of those, not all of them."""
    if isinstance(cand, (list, tuple, set)):
        return any(_op_contains(cand, v, substring_for_list=substring_for_list) for v in values)
    return any(_op_equals(cand, v) for v in values)


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
                out[name.lower()] = dict(meta)
            else:  # {"Python": 5}
                out[name.lower()] = {"years": meta}
    elif isinstance(raw, (list, tuple)):
        out = {str(s).lower(): {"years": None} for s in raw}
    else:
        out = {}
    # Real per-skill years inferred from resume text (see
    # candidates._extract_skill_years) -- overlays the default None above so
    # skill_experience filters can actually resolve instead of always
    # failing on absent data. A skill this candidate has but never
    # mentioned in any experience's description keeps years=None (unknown),
    # not 0 (verified none) -- see matches_filter's handling of None.
    for skill, years in (candidate.get("skill_years") or {}).items():
        out.setdefault(skill.lower(), {})["years"] = years
    return out


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
        if f.operator in {"in", "not_in"} and isinstance(value, (list, tuple, set)):
            hit = _op_in(value, f.value, substring_for_list=f.field in _FREE_TEXT_LIST_FIELDS)
            return hit if f.operator == "in" else not hit
        return op(value, f.value)
    except (ValueError, TypeError):
        return False


def _alternative_group_matches(candidate: dict, group) -> bool:
    """A candidate satisfies an AlternativeGroup if AT LEAST ONE branch's
    filters ALL match (AND within a branch, OR across branches) -- see
    AlternativeGroup's docstring in schemas.py. An empty branch (no
    filters) is vacuously satisfied, same as an empty overall filter list
    elsewhere in this engine."""
    return any(
        all(matches_filter(candidate, f) for f in branch)
        for branch in group.branches
    )


def apply_spec(candidates: list[dict], spec: FilterSpec) -> list[dict]:
    """Apply the full filter spec with AND/OR/NOT logic. Scores untouched.

    `spec.alternative_groups` (see AlternativeGroup's docstring) is always
    ANDed on top of the result of `spec.filters`/`spec.logic` -- a
    candidate must pass the ordinary filters AND satisfy every alternative
    group, regardless of what `spec.logic` is for the ordinary filters."""
    if not spec.filters and not spec.alternative_groups:
        return list(candidates)

    kept = []
    for c in candidates:
        if spec.filters:
            checks = [matches_filter(c, f) for f in spec.filters]
            if spec.logic == "OR":
                keep = any(checks)
            elif spec.logic == "NOT":
                keep = not any(checks)
            else:  # AND
                keep = all(checks)
        else:
            keep = True
        if keep and spec.alternative_groups:
            keep = all(_alternative_group_matches(c, g) for g in spec.alternative_groups)
        if keep:
            kept.append(c)

    # Preserve match-score ordering (highest first). Original scores unchanged.
    return sorted(kept, key=lambda c: c.get("match_score", 0), reverse=True)
