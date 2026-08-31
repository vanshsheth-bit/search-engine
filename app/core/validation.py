"""Validation layer — the safety gate between the LLM and the engine.

Even with schema-constrained decoding, we re-validate everything here:
- field exists in vocabulary
- operator is legal for that field's value type
- value type matches the field
- skill-scoped fields carry a `skill`
- the candidate dataset actually has the field (no inventing attributes)
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

from app.core.vocabulary import (
    ALLOWED_OPERATORS,
    FIELD_LABELS,
    FIELD_TYPES,
    GENERIC_FILLER_WORDS,
    NAME_FIELDS,
    OPERATORS_BY_TYPE,
    SKILL_SCOPED_FIELDS,
)
from app.models.schemas import Filter


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
        label = FIELD_LABELS.get(f.field, f.field)

        if f.field not in FIELD_TYPES:
            skipped.append(f"\"{f.field}\" isn't something available for these candidates")
            any_unsupported = True
            continue

        expected_type = FIELD_TYPES[f.field]

        if f.operator not in ALLOWED_OPERATORS:
            skipped.append(f"I didn't understand the comparison for {label}")
            continue

        if f.operator not in OPERATORS_BY_TYPE.get(expected_type, set()):
            skipped.append(f"that comparison doesn't make sense for {label}")
            continue

        if f.field in SKILL_SCOPED_FIELDS and not f.skill:
            skipped.append(
                "I need to know which skill a years-of-experience filter applies to"
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

        f = _coerce_value(f, expected_type)

        if not _type_ok(f.value, expected_type):
            skipped.append(f"that doesn't look like a valid value for {label}")
            continue

        if available_fields is not None:
            probe = "skill" if f.field in {"skill", "skill_experience"} else f.field
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
