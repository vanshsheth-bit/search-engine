"""Single source of truth for the filter vocabulary.

Everything downstream (LLM prompt, JSON schema, validation, engine) derives
from these constants so the system can never drift out of sync.
"""
from __future__ import annotations

import re

# Fields a recruiter may filter on. Map each to the expected value type so
# the validator can reject type mismatches deterministically.
FIELD_TYPES: dict[str, str] = {
    "location": "string",             # a specific city
    "country": "string",              # country-level, from real gazetteer data (not city-level)
    "experience": "number",          # total years of experience
    "skill": "string",               # a single skill name (contains check)
    "skill_experience": "number",    # years for a *specific* skill
    "education": "ordinal",          # degree level, e.g. "Bachelor", "Master"
    "university": "string",          # university/college name(s) attended
    "college_tier": "ordinal",       # Low / Medium / High, from a ranking dataset
    "company": "string",             # company name(s) worked at
    "company_tier": "ordinal",       # Low / Medium / High, from a ranking dataset
    "notice_period": "number",       # numeric, paired with a unit
    "relocation": "boolean",         # willing to relocate
    "job_title": "string",           # position/role title(s) held
    "certification": "string",       # certification name(s), free-text (contains check)
    "employment_gap_months": "number",  # longest single employment gap, in months
    "company_type": "string",        # Product / Service / Both, from cached LLM classification
    "domain": "string",              # industry/functional domain (e.g. "FinTech", "Healthcare
                                      # IT Engineering"), classified from real experience text
                                      # -- see experience_index/classifications.jsonl
}

ALLOWED_FIELDS: list[str] = list(FIELD_TYPES.keys())

# Human-readable labels for error/clarify messages -- recruiters never see
# raw field keys like "skill_experience".
FIELD_LABELS: dict[str, str] = {
    "location": "location",
    "country": "country",
    "experience": "years of experience",
    "skill": "skill",
    "skill_experience": "years of experience with a specific skill",
    "education": "education level",
    "university": "university/college attended",
    "college_tier": "college ranking/tier",
    "company": "company worked at",
    "company_tier": "company ranking/tier",
    "notice_period": "notice period",
    "relocation": "willingness to relocate",
    "job_title": "job title/role held",
    "certification": "certification",
    "employment_gap_months": "longest employment gap (months)",
    "company_type": "company type (product/service)",
    "domain": "industry/functional domain",
}

# Fields that MUST carry a `skill` key (which skill the number refers to).
SKILL_SCOPED_FIELDS = {"skill_experience"}

NUMERIC_OPERATORS = {"gte", "lte", "gt", "lt"}
STRING_OPERATORS = {"equals", "not_equals", "contains", "not_contains"}
LIST_OPERATORS = {"in", "not_in"}
BOOLEAN_OPERATORS = {"equals", "not_equals"}

ALLOWED_OPERATORS: list[str] = sorted(
    NUMERIC_OPERATORS | STRING_OPERATORS | LIST_OPERATORS
)

ALLOWED_LOGIC = {"AND", "OR", "NOT"}

# Which operators are valid for which value type. Used by the validator.
OPERATORS_BY_TYPE: dict[str, set[str]] = {
    "string": STRING_OPERATORS | LIST_OPERATORS,
    "number": NUMERIC_OPERATORS | {"equals", "not_equals"} | LIST_OPERATORS,
    "boolean": BOOLEAN_OPERATORS,
    "ordinal": NUMERIC_OPERATORS | {"equals", "not_equals"} | LIST_OPERATORS,
}

# Degree-level ranking, matched by whole-word keyword against the text
# rather than exact string identity -- so "Masters", "Master's", "MS",
# "M.Tech" and "Master" all resolve to the same rank instead of silently
# failing to match on phrasing/pluralization. Word-boundary matching (not
# plain substring) avoids false positives like "Systems" containing "ms".
_DEGREE_KEYWORDS: list[tuple[int, tuple[str, ...]]] = [
    # "doctor of philosophy" is the single most common full-length way a
    # PhD is actually written on a resume/transcript -- confirmed missing
    # here (returned no rank at all, not even ranked below a Bachelor's)
    # until this was added, silently breaking a real "PhD + Bachelor's"
    # candidate's degree-level down to just "Bachelor".
    (5, ("phd", "doctorate", "doctorates", "doctor of philosophy", "dphil", "doctoral")),
    (4, ("master", "masters", "mba", "mtech", "msc", "ms")),
    (3, ("bachelor", "bachelors", "btech", "bsc", "bs")),
    (2, ("diploma", "diplomas", "associate", "associates")),
    (1, ("high school",)),
]

EDUCATION_RANK_LABELS: dict[int, str] = {
    5: "Doctorate", 4: "Master", 3: "Bachelor", 2: "Diploma", 1: "High School",
}

# Back-compat plain lookup (exact canonical labels only). Prefer
# `education_rank()` for anything that has to handle real-world phrasing.
EDUCATION_RANK = {label.lower(): rank for rank, label in EDUCATION_RANK_LABELS.items()}
EDUCATION_RANK["phd"] = 5


def _normalize_degree_text(text: str) -> str:
    return re.sub(r"[.'’]", "", text.lower())


def education_rank(text: str | None) -> int | None:
    """Canonical ordinal rank for a degree string. Robust to phrasing --
    'Masters', "Master's", 'MS', 'M.Tech', 'Master' all resolve to the same
    rank -- so equals/gte/lte comparisons never fail on wording alone."""
    if not text:
        return None
    norm = _normalize_degree_text(str(text).strip())
    for rank, keywords in _DEGREE_KEYWORDS:
        for kw in keywords:
            pattern = kw if " " in kw else rf"\b{re.escape(kw)}\b"
            if re.search(pattern, norm):
                return rank
    return None


def bare_degree_rank(text: str | None) -> int | None:
    """Rank ONLY if `text`, once stripped of filler words ("degree",
    "level", "holder"), IS ENTIRELY a degree-level keyword phrase (e.g.
    "PhD", "bachelor's degree") -- not just contains one as a substring of
    some longer, unrelated real value. Deliberately stricter than
    education_rank (which scans for a keyword anywhere in longer resume
    text, e.g. a job description) -- this is for self-healing a filter
    value the LLM put in the WRONG field (see validation.py): confirmed
    live, "PhD" and "bachelor degree" both got routed into `job_title`
    instead of `education`. Auto-correcting that is only safe because the
    value is nothing BUT the degree word -- "PhD Research Manager" is a
    real, different job title that happens to mention a degree, and must
    NOT be reinterpreted as an education filter."""
    if not text or not isinstance(text, str):
        return None
    norm = _normalize_degree_text(text.strip())
    norm = re.sub(r"\s*(degrees?|levels?|holders?)\s*$", "", norm).strip()
    for rank, keywords in _DEGREE_KEYWORDS:
        if norm in keywords:
            return rank
    return None


# Shared Low/Medium/High tier scale, used by both college_tier (from
# master_universities.csv) and company_tier (from company_ranks.json).
# Ranked, not just labeled, so "top tier" can mean "gte High" the same way
# degree-level queries do.
_TIER_RANK = {"low": 1, "medium": 2, "high": 3}
TIER_LABELS = {1: "Low", 2: "Medium", 3: "High"}


def tier_rank(text: str | None) -> int | None:
    if not text:
        return None
    return _TIER_RANK.get(str(text).strip().lower())


# Both fields share one rank scale -- one function, two names for clarity
# at call sites.
college_tier_rank = tier_rank
company_tier_rank = tier_rank


# Fields whose value should be a specific proper name (a university, a
# company) -- used to catch a real failure mode where a weak model turns a
# generic qualifier ("good universities") into a literal name search
# ("university contains 'good'") instead of routing it to the tier field.
NAME_FIELDS = {"university", "company"}
GENERIC_FILLER_WORDS = {
    "good", "great", "top", "best", "nice", "decent", "solid", "strong",
    "known", "famous", "big", "large", "small", "prestigious", "reputed",
    "reputable", "quality", "any", "some", "a", "an", "the",
}


VALID_INTENTS = {
    "FILTER_CANDIDATES", "CLARIFY", "UNSUPPORTED_FILTER", "LOOKUP",
    "EXPERIENCE_SEARCH",
}
