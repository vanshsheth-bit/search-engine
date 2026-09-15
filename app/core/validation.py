"""Validation layer — the safety gate between the LLM and the engine.

Even with schema-constrained decoding, we re-validate everything here:
- field exists in vocabulary
- operator is legal for that field's value type
- value type matches the field
- skill-scoped fields carry a `skill`
- the candidate dataset actually has the field (no inventing attributes)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field

from app.core.candidates import canonicalize_country
from app.core.skill_taxonomy import canonicalize, expand_skill_term
from app.core.vocabulary import (
    ALLOWED_OPERATORS,
    EDUCATION_RANK_LABELS,
    FIELD_LABELS,
    FIELD_TYPES,
    GENERIC_FILLER_WORDS,
    GENERIC_SKILL_FILLER_WORDS,
    NAME_FIELDS,
    OPERATORS_BY_TYPE,
    SKILL_SCOPED_FIELDS,
    bare_degree_rank,
    education_rank,
)
from app.models.schemas import AlternativeGroup, Filter


@dataclass
class ValidationResult:
    ok: bool
    filters: list[Filter] = dc_field(default_factory=list)
    error: str | None = None
    unsupported: bool = False  # True -> respond with UNSUPPORTED_FILTER
    # Per-filter reasons for filters that were dropped but did NOT abort the
    # whole request -- e.g. one unsupported clause in an otherwise-valid
    # compound query. Surfaced to the recruiter alongside real results rather
    # than silently swallowed, so "8+ years, Kubernetes, and relocating" still
    # returns matches for the two real filters instead of nothing.
    skipped: list[str] = dc_field(default_factory=list)


def _coerce_value(f: Filter, expected_type: str) -> Filter:
    """Best-effort coercion so '3' becomes 3 for numeric fields."""
    if expected_type == "number" and not isinstance(f.value, list):
        try:
            num = float(f.value)
            f.value = int(num) if num.is_integer() else num
        except (TypeError, ValueError):
            pass
    if expected_type == "boolean" and isinstance(f.value, str):
        low = f.value.strip().lower()
        if low in {"true", "yes", "1"}:
            f.value = True
        elif low in {"false", "no", "0"}:
            f.value = False
    return f


def _type_ok(value, expected_type: str) -> bool:
    if isinstance(value, list):
        return True  # in / not_in lists
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return isinstance(value, str)


# Deterministic self-healing for a confirmed, reproducible LLM routing
# mistake: "contains"/"not_contains" on an ordinal field (education,
# college_tier, company_tier) -- e.g. "PhD candidates in Mumbai" or "bachelor
# degree" parsed as {"field":"education","operator":"contains","value":"PhD"}
# instead of "gte". Confirmed this is NOT occasional model flakiness --
# reproduced 100% of the time even with a freshly cold model cache and a
# directly-matching worked few-shot example already in the prompt (see
# prompt.py's PhD/bachelor examples right after rule 5b) -- so prompt
# engineering alone doesn't reliably fix it here, and it's corrected
# deterministically instead. Safe to auto-remap rather than just drop the
# filter: rule 5b's own logic already establishes that a bare degree/tier
# mention with no "only"/"exactly" qualifier means "at least" that level --
# what "contains" degrades to here isn't a guess about the recruiter's
# intent, it's a syntax-level fix for a known, mechanical mistake.
_ORDINAL_CONTAINS_FIX = {"contains": "gte", "not_contains": "not_equals"}

# Deterministic self-healing for a third confirmed, reproducible LLM routing
# mistake: "<Skill> developer" ("Python developer", "React developer", ...)
# routed as ONE literal job_title phrase instead of decomposing into the real
# skill. Confirmed on real data: 0 of 103 real candidates in this dataset
# have "python developer" (or any tech name + "developer") as a literal job
# title, while 40 of them have Python as a real, declared skill -- "<tech>
# developer" describes what someone builds WITH, not a title anyone actually
# holds verbatim; real titles here are things like "Software Developer",
# "Data Analyst", "Systems Engineer". Also confirmed the naive fix of just
# ALSO requiring job_title contains "developer" alongside the skill is worse,
# not better: doing so drops the same query from 40 real matches to 16,
# since most Python users here are titled something else entirely.
#
# Deliberately scoped to "developer"/"dev" only, NOT "engineer" -- "<X>
# Engineer" (DevOps Engineer, ML Engineer, Data Engineer, QA Engineer, Site
# Reliability Engineer) is overwhelmingly an ESTABLISHED, standalone job
# title convention in its own right, not a generic-role-noun description of
# "does X" -- collapsing those into a bare skill filter would be wrong in
# the opposite direction, and "engineer" carries no comparable "nobody is
# ever actually titled this" property the way "developer" does here.
#
# Only fires when the remaining phrase is a REAL, taxonomy-recognized tool
# (skill_taxonomy.expand_skill_term returns None for anything the taxonomy
# doesn't cover) -- confirmed safe against the real taxonomy: "software",
# "web", "backend", "frontend", "business", "senior", "devops", "ios", and
# "android" are all NOT recognized tool entries, so "Software Developer",
# "Business Developer", "iOS Developer" etc. correctly fall through
# untouched (real, distinct titles/roles, not a skill+generic-noun mislabel)
# while "python", "java", "react", "kubernetes" etc. are.
#
# Falls back to just the word immediately before the role noun (e.g.
# "Senior Python Developer" -> "python") when the full prefix isn't
# recognized as one unit -- a leading modifier (seniority, "lead", "junior")
# is common phrasing this shouldn't refuse just because it isn't part of
# the tool's own name.
_GENERIC_DEV_SUFFIXES = {"developer", "developers", "dev", "devs"}


def _skill_developer_phrase(value) -> str | None:
    """Returns the real skill name to heal a "<tech> developer"-shaped
    job_title value into, or None if `value` doesn't match that shape at all
    -- most job titles don't, and this must stay conservative (see
    _GENERIC_DEV_SUFFIXES' docstring)."""
    if not isinstance(value, str):
        return None
    words = value.strip().split()
    if len(words) < 2:
        return None
    if words[-1].strip(".").lower() not in _GENERIC_DEV_SUFFIXES:
        return None
    prefix_words = words[:-1]
    prefix = " ".join(prefix_words).strip()
    if not prefix:
        return None
    if expand_skill_term(prefix) is not None:
        return canonicalize(prefix)
    tail = prefix_words[-1]
    if expand_skill_term(tail) is not None:
        return canonicalize(tail)
    return None


# Common recruiter shorthand for a job-title word, expanded to what a real
# resume actually spells out -- confirmed live: "backend engg" (recruiter's
# own abbreviation for "engineer") searched as a literal job_title substring
# matched 0 candidates on a 99-person pool that has real "Backend Engineer"
# titles, because nobody's stored title literally contains "engg". Word-
# boundary, case-insensitive, whole-word only (never a substring match
# inside a longer word) -- "sr" must not touch "Senior" already spelled out,
# and must not fire inside an unrelated word that happens to contain these
# letters.
_TITLE_ABBREVIATIONS = {
    "engg": "Engineer", "eng": "Engineer",
    "mgr": "Manager", "mgmt": "Management",
    "sr": "Senior", "jr": "Junior",
    "dev": "Developer", "devs": "Developers",
    "admin": "Administrator", "arch": "Architect",
    "tech": "Technical", "asst": "Assistant",
    "exec": "Executive", "coord": "Coordinator",
    "spec": "Specialist", "eng.": "Engineer",
}
_TITLE_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(
        _TITLE_ABBREVIATIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def _expand_title_abbreviations(value) -> str | None:
    """Expands every recognized abbreviation in a job_title value to its
    real word, or None if `value` contains none (so the caller can tell
    "nothing to change" apart from "changed to the same text")."""
    if not isinstance(value, str):
        return None
    expanded = _TITLE_ABBREV_RE.sub(
        lambda m: _TITLE_ABBREVIATIONS[m.group(0).lower()], value,
    )
    return expanded if expanded != value else None


_NOT_OPERATORS = {"not_contains", "not_equals", "not_in"}


def validate_filters(
    filters: list[Filter],
    available_fields: set[str] | None = None,
) -> ValidationResult:
    """Validate a list of filters. `available_fields` is the set of fields the
    candidate dataset actually provides; if None, all vocabulary fields are
    assumed available.

    Each filter is checked independently: a bad one is dropped and its reason
    recorded in `skipped`, but does NOT abort the rest of the batch -- a
    compound query like "8+ years, knows Kubernetes, and open to relocating"
    should still return real matches for the two valid filters, with a note
    that relocation couldn't be checked, rather than failing the whole
    request over one unsupported clause. Only when NOTHING survives does the
    overall result come back not-ok (unsupported takes priority in that case
    if any drop reason was "not available" rather than "malformed")."""
    validated: list[Filter] = []
    skipped: list[str] = []
    any_unsupported = False

    for f in filters:
        # Deterministic self-healing for a second confirmed, reproducible
        # LLM routing mistake: a BARE degree word/phrase ("PhD", "bachelor
        # degree") routed into job_title instead of education -- same
        # class of mechanical mistake as the ordinal-operator fix below,
        # confirmed live. Only fires when the value is nothing BUT the
        # degree word (see bare_degree_rank's docstring) -- a real title
        # that happens to mention a degree ("PhD Research Manager") is
        # left untouched.
        if f.field == "job_title" and f.operator in {"contains", "equals"}:
            rank = bare_degree_rank(f.value)
            if rank is not None:
                f.field = "education"
                f.operator = "gte"
                f.value = EDUCATION_RANK_LABELS[rank]

        if f.field == "job_title" and f.operator in {"contains", "not_contains", "equals", "not_equals"}:
            expanded_title = _expand_title_abbreviations(f.value)
            if expanded_title is not None:
                f.value = expanded_title

        if f.field == "job_title" and f.operator in {"contains", "not_contains", "equals", "not_equals"}:
            skill_value = _skill_developer_phrase(f.value)
            if skill_value is not None:
                f.field = "skill"
                f.operator = "not_contains" if f.operator in _NOT_OPERATORS else "contains"
                f.value = skill_value

        # Deterministic self-healing for a fourth confirmed, reproducible LLM
        # routing mistake: a COUNTRY abbreviation ("UAE", "USA", "UK") routed
        # into "location" (a specific CITY field) instead of "country" --
        # confirmed live in the eval harness: "engineers based in the UAE"
        # produced {"field":"location","value":"UAE"}, which can never match
        # any real candidate (locations are stored as real city names, never
        # a bare country abbreviation) even though this exact query's own
        # worked example elsewhere in the prompt gets it right. Reuses the
        # SAME alias table candidates.py already applies to a real "country"
        # filter (_COUNTRY_ALIASES) -- canonicalize_country returns the value
        # UNCHANGED for anything it doesn't recognize, so this only fires for
        # the small set of KNOWN country abbreviations, never a real city
        # name that merely happens to be a short word.
        if f.field == "location" and isinstance(f.value, str):
            resolved_country = canonicalize_country(f.value)
            if resolved_country != f.value:
                f.field = "country"
                f.value = resolved_country

        label = FIELD_LABELS.get(f.field, f.field)

        if f.field not in FIELD_TYPES:
            skipped.append(f"\"{f.field}\" isn't something available for these candidates")
            any_unsupported = True
            continue

        expected_type = FIELD_TYPES[f.field]

        if expected_type == "ordinal" and f.operator in _ORDINAL_CONTAINS_FIX:
            f.operator = _ORDINAL_CONTAINS_FIX[f.operator]

        if f.operator not in ALLOWED_OPERATORS:
            skipped.append(f"I didn't understand the comparison for {label}")
            continue

        if f.operator not in OPERATORS_BY_TYPE.get(expected_type, set()):
            skipped.append(f"that comparison doesn't make sense for {label}")
            continue

        if f.field in SKILL_SCOPED_FIELDS and not f.skill:
            skipped.append(
                "I need to know which skill or domain a years-of-experience filter applies to"
            )
            continue

        if (
            f.field in NAME_FIELDS
            and isinstance(f.value, str)
            and f.value.strip().lower() in GENERIC_FILLER_WORDS
        ):
            tier_field = "college_tier" if f.field == "university" else "company_tier"
            skipped.append(
                f"\"{f.value}\" doesn't look like a specific {label} name -- "
                f"did you mean a ranking ({FIELD_LABELS[tier_field]}) instead?"
            )
            continue

        # Same class of mistake as the NAME_FIELDS check above, for "skill"
        # instead of university/company -- confirmed live: the bare query
        # "Skills" (naming no specific technology at all) got parsed as a
        # literal search for a skill named "Skills". Caught here, BEFORE
        # the fuzzy-matching pipeline ever sees it -- that pipeline embeds
        # whatever term survives validation and asks an LLM judge whether
        # candidates' real skills "satisfy" it, with no way of its own to
        # recognize the term itself names nothing. Dropping it here, same
        # as any other invalid filter, is what keeps a meaningless term
        # from ever reaching that expensive, confident-looking-but-wrong
        # path at all.
        if (
            f.field == "skill"
            and isinstance(f.value, str)
            and f.value.strip().lower() in GENERIC_SKILL_FILLER_WORDS
        ):
            skipped.append(
                f"\"{f.value}\" isn't a specific skill or technology -- which one did you mean?"
            )
            continue

        # Real, reported live bug: a vague education phrase with no stated
        # degree ("done high education from...") made the model invent
        # `{"field":"education","operator":"gte","value":"High"}` --
        # "High"/"Low" are real values for the ordinal TIER fields
        # (college_tier/company_tier), not degree labels, and
        # education_rank("High") returns None: this filter could never
        # match any real candidate (their `education` is a degree name,
        # never compared against a rank that doesn't exist) yet passed
        # through silently, with no note explaining the empty result.
        # Caught here the same way an unrecognized skill term is, before it
        # ever reaches engine.py's comparison.
        if f.field == "education" and f.operator in {"gte", "lte", "equals", "not_equals"}:
            if education_rank(f.value) is None:
                skipped.append(
                    f"\"{f.value}\" isn't a recognized education level "
                    f"(e.g. Bachelor's, Master's, Doctorate) -- which one did you mean?"
                )
                continue

        # notice_period's engine matching never actually reads `unit` (see
        # engine.py -- it's compared as a plain number of days) and
        # merge.chip_label already defaults a missing one to "days" for
        # display, but the LLM output itself should carry it too rather than
        # relying on that display-only fallback -- confirmed live the model
        # sometimes omits it even on an otherwise-correct filter. "days" is
        # the overwhelmingly common case and what every other layer already
        # assumes, so default it here rather than leaving it unset.
        if f.field == "notice_period" and not f.unit:
            f.unit = "days"

        f = _coerce_value(f, expected_type)

        if not _type_ok(f.value, expected_type):
            skipped.append(f"that doesn't look like a valid value for {label}")
            continue

        if available_fields is not None:
            if f.field in {"skill", "skill_experience"}:
                probe = "skill"
            elif f.field == "domain_experience":
                probe = "domain"
            else:
                probe = f.field
            if probe not in available_fields:
                skipped.append(f"I don't have {label} data for these candidates")
                any_unsupported = True
                continue

        validated.append(f)

    if not validated and skipped:
        msg = "; ".join(skipped)
        msg = msg[0].upper() + msg[1:]
        if not msg.endswith((".", "?", "!")):
            msg += "."
        return ValidationResult(ok=False, unsupported=any_unsupported, error=msg)

    return ValidationResult(ok=True, filters=validated, skipped=skipped)


def validate_alternative_groups(
    groups: list[AlternativeGroup],
    available_fields: set[str] | None = None,
) -> tuple[list[AlternativeGroup], list[str]]:
    """Validates each group's filters via validate_filters, unchanged above.
    Unlike the top-level flat list (where one bad clause among independent
    AND'd filters is safely dropped alone -- see validate_filters' own
    docstring), a group's filters are logically coupled: they together form
    ONE eligibility route. Silently dropping one leaf out of a route would
    silently WEAKEN what the route means (e.g. "Master's from a Tier-1
    university" degrading to just "Master's" if the college_tier leaf
    failed validation) -- a worse silent failure than dropping the whole
    route. So: if ANY leaf in a group fails validation, the ENTIRE group is
    dropped, with a note explaining why. If every group ends up dropped,
    the caller is expected to drop alternative_groups entirely -- the rest
    of the flat query still applies; this function never fails the whole
    request over it, same "a bad piece is dropped, doesn't abort the rest"
    philosophy as validate_filters.

    Every surviving leaf is forced hard=True regardless of what was set --
    see AlternativeGroup's docstring: a soft preference has no meaning
    inside an eligibility route."""
    validated_groups: list[AlternativeGroup] = []
    notes: list[str] = []
    for group in groups:
        if not group.filters:
            continue
        result = validate_filters(list(group.filters), available_fields)
        if not result.ok or len(result.filters) != len(group.filters):
            reason = result.error or "; ".join(result.skipped) or "one of its requirements isn't supported"
            notes.append(f"Dropped an alternative requirement route ({reason})")
            continue
        forced = [f.model_copy(update={"hard": True}) for f in result.filters]
        validated_groups.append(AlternativeGroup(filters=forced))
    return validated_groups, notes
