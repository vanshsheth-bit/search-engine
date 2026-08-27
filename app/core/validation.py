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
    assumed available."""
    validated: list[Filter] = []

    for f in filters:
        label = FIELD_LABELS.get(f.field, f.field)

        if f.field not in FIELD_TYPES:
            return ValidationResult(
                ok=False, unsupported=True,
                error=f"I can't filter on \"{f.field}\" -- that's not something "
                      f"available for these candidates.",
            )

        expected_type = FIELD_TYPES[f.field]

        if f.operator not in ALLOWED_OPERATORS:
            return ValidationResult(
                ok=False,
                error="Sorry, I didn't understand that comparison -- could you "
                      "rephrase it?",
            )

        if f.operator not in OPERATORS_BY_TYPE.get(expected_type, set()):
            return ValidationResult(
                ok=False,
                error=f"That comparison doesn't make sense for {label} -- "
                      f"could you rephrase?",
            )

        if f.field in SKILL_SCOPED_FIELDS and not f.skill:
            return ValidationResult(
                ok=False,
                error="I need to know which skill that years-of-experience "
                      "applies to -- try phrasing it like \"5+ years of "
                      "Python\", or if you meant overall experience, \"5+ "
                      "years of experience\".",
            )

        if (
            f.field in NAME_FIELDS
            and isinstance(f.value, str)
            and f.value.strip().lower() in GENERIC_FILLER_WORDS
        ):
            tier_field = "college_tier" if f.field == "university" else "company_tier"
            return ValidationResult(
                ok=False,
                error=f"\"{f.value}\" doesn't look like a specific {label} name -- "
                      f"could you name the actual one, or did you mean a ranking "
                      f"(e.g. \"top tier\")? That's a different field ({FIELD_LABELS[tier_field]}).",
            )

        f = _coerce_value(f, expected_type)

        if not _type_ok(f.value, expected_type):
            return ValidationResult(
                ok=False,
                error=f"That doesn't look like a valid value for {label} -- "
                      f"could you rephrase?",
            )

        if available_fields is not None:
            probe = "skill" if f.field in {"skill", "skill_experience"} else f.field
            if probe not in available_fields:
                return ValidationResult(
                    ok=False, unsupported=True,
                    error=f"I don't have {label} data for these candidates, "
                          f"so I can't filter on that.",
                )

        validated.append(f)

    return ValidationResult(ok=True, filters=validated)
