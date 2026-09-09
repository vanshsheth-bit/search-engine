"""Merge new filters into existing session state, and render chips."""
from __future__ import annotations

from app.models.schemas import AlternativeGroup, Chip, Filter

_OP_SYMBOL = {"gte": "≥", "lte": "≤", "gt": ">", "lt": "<"}
_FIELD_ICON = {
    "location": "📍",
    "country": "🌍",
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
    "job_title": "💼",
    "certification": "📜",
    "employment_gap_months": "🕳️",
    "company_type": "🏭",
    "domain": "🏦",
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
        prefix = "No " if f.operator in {"not_contains", "not_in"} else ""
        # A skill concept expanded via the taxonomy (e.g. "machine learning"
        # -> its real tools) carries a LIST here, not a single string --
        # show the representative (first/canonical) term plus a count rather
        # than a raw Python list repr.
        if isinstance(f.value, list):
            label = str(f.value[0]) if f.value else ""
            if len(f.value) > 1:
                label += f" +{len(f.value) - 1} more"
        else:
            label = f.value
        return f"{icon} {prefix}{label}".strip()
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
    if f.field == "company_type":
        label = "/".join(f.value) if isinstance(f.value, list) else f.value
        prefix = "Not " if f.operator in {"not_contains", "not_in", "not_equals"} else ""
        return f"{icon} {prefix}{label}-based".strip()
    if f.field == "job_title":
        prefix = "Not " if f.operator in {"not_contains", "not_equals", "not_in"} else ""
        label = " or ".join(f.value) if isinstance(f.value, list) else f.value
        return f"{icon} {prefix}{label}".strip()
    if f.field == "certification":
        prefix = "No " if f.operator in {"not_contains", "not_equals"} else ""
        return f"{icon} {prefix}{f.value}".strip()
    if f.field == "employment_gap_months":
        sym = _OP_SYMBOL.get(f.operator, "")
        return f"{icon} Gap {sym} {f.value} mo".strip()
    prefix = "Not " if f.operator in {"not_equals", "not_contains", "not_in"} else ""
    # Generic fallback for any other field (e.g. domain, location) -- an
    # "in" filter carries a list of alternative values here; show them
    # joined rather than a raw Python list repr.
    label = "/".join(f.value) if isinstance(f.value, list) else f.value
    return f"{icon} {prefix}{label}".strip()


def to_chips(filters: list[Filter]) -> list[Chip]:
    return [
        Chip(label=chip_label(f), field=f.field, skill=f.skill) for f in filters
    ]


def preferred_chips(filters: list[Filter]) -> list[Chip]:
    """See LLMOutput.preferred_filters -- rendered visually distinct
    (`preferred=True`) since these never exclude a candidate, only rank
    matches within what already qualified."""
    return [
        Chip(label=f"☆ {chip_label(f)}", field=f.field, skill=f.skill, preferred=True)
        for f in filters
    ]


def alternative_group_chips(groups: list[AlternativeGroup]) -> list[Chip]:
    """One combined chip per group, e.g. "Master's (Tier-1) OR ≥ 10 yrs" --
    a group has no single `field` of its own (it can span several), so
    `field` is left as a synthetic marker rather than a real vocabulary
    field name."""
    chips = []
    for group in groups:
        branch_labels = [
            " + ".join(chip_label(f) for f in branch) for branch in group.branches if branch
        ]
        if not branch_labels:
            continue
        chips.append(Chip(
            label=" OR ".join(branch_labels), field="_alternative_group", alternative=True,
        ))
    return chips
