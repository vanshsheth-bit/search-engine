"""Merge new filters into existing session state, and render chips."""
from __future__ import annotations

from app.models.schemas import Chip, Filter

_OP_SYMBOL = {"gte": "≥", "lte": "≤", "gt": ">", "lt": "<"}
_FIELD_ICON = {
    "location": "📍",
    "experience": "🧭",
    "skill": "🧩",
    "skill_experience": "🧩",
    "education": "🎓",
    "university": "🏫",
    "college_tier": "🏆",
    "company": "🏢",
    "company_tier": "🏆",
    "notice_period": "⏱️",
    "relocation": "✈️",
}


def merge_filters(existing: list[Filter], incoming: list[Filter]) -> list[Filter]:
    """Incoming filters replace existing ones with the same key (field+skill).
    This is what makes 'actually, Bangalore instead' update rather than
    duplicate the location filter."""
    merged: dict[tuple, Filter] = {f.key(): f for f in existing}
    for f in incoming:
        merged[f.key()] = f
    return list(merged.values())


def chip_label(f: Filter) -> str:
    icon = _FIELD_ICON.get(f.field, "🔖")
    if f.field == "skill_experience":
        sym = _OP_SYMBOL.get(f.operator, "")
        return f"{icon} {f.skill} {sym} {f.value} yrs".strip()
    if f.field == "experience":
        sym = _OP_SYMBOL.get(f.operator, "")
        return f"{icon} Experience {sym} {f.value} yrs".strip()
    if f.field == "notice_period":
        sym = _OP_SYMBOL.get(f.operator, "")
        unit = f.unit or "days"
        return f"{icon} Notice {sym} {f.value} {unit}".strip()
    if f.field == "skill":
        prefix = "No " if f.operator in {"not_contains", "not_equals"} else ""
        return f"{icon} {prefix}{f.value}".strip()
    if f.field == "relocation":
        return f"{icon} Willing to relocate"
    if f.field == "college_tier":
        sym = _OP_SYMBOL.get(f.operator, "")
        return f"{icon} {sym} {f.value} tier".strip()
    if f.field == "university":
        prefix = "Not from " if f.operator in {"not_contains", "not_equals"} else ""
        return f"{icon} {prefix}{f.value}".strip()
    if f.field == "company_tier":
        sym = _OP_SYMBOL.get(f.operator, "")
        return f"{icon} {sym} {f.value} tier company".strip()
    if f.field == "company":
        prefix = "Not at " if f.operator in {"not_contains", "not_equals"} else ""
        return f"{icon} {prefix}{f.value}".strip()
    prefix = "Not " if f.operator in {"not_equals", "not_contains", "not_in"} else ""
    return f"{icon} {prefix}{f.value}".strip()


def to_chips(filters: list[Filter]) -> list[Chip]:
    return [
        Chip(label=chip_label(f), field=f.field, skill=f.skill) for f in filters
    ]
