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
    "domain_experience": "🏦",
}


# Years-based fields where "which one wins on conflict" should mean
# "whichever is the stricter, more-experienced requirement", not "whichever
# was stated most recently" -- see _stronger_experience_filter's docstring
# for the real, reported case this fixes. Deliberately narrow: tier/ordinal
# fields (company_tier, education, ...) and non-experience numeric fields
# (notice_period, employment_gap_months) keep the plain last-write-wins
# behavior below, since "actually, 30 days notice instead of 60" is a
# genuine override the recruiter stated on purpose, not stale carryover.
_EXPERIENCE_YEAR_FIELDS = {"experience", "skill_experience", "domain_experience"}


def _stronger_experience_filter(old: Filter, new: Filter) -> Filter:
    """Real, reported live bug: a flat `experience >= 0` left over from an
    earlier, unrelated turn survived untouched (different mechanism, same
    key never gets re-touched) right alongside a NEW turn's much stronger
    "senior" requirement, displaying as a nonsensical "Experience >= 0 yrs"
    chip alongside "(Experience >= 7 AND <= 12) OR Senior title" -- because
    plain last-write-wins only fires when the SAME key is re-stated, and
    even when it does fire (e.g. two `experience gte` values across turns,
    with nothing in between to explain why one should beat the other), origin
    order is a meaningless tie-breaker for "how many years of experience is
    actually required" -- picking the answer that asks for MORE experience
    is always at least as correct as picking the one that just happens to be
    newest, and strictly safer against silently under-matching after a stale
    filter's carried-over minimum quietly override a just-stated stronger one.

    Only reconciles two filters that are directionally comparable (same
    operator, a plain numeric value on both sides) -- `gte` keeps the LARGER
    value (require at least as many years as the stricter of the two asks),
    `lte` keeps the SMALLER value (the tighter of the two ceilings), by the
    same "stricter wins" principle mirrored in the other direction. Anything
    else (mismatched operators, gt/lt, a non-numeric value) has no single
    correct comparison -- falls through to the caller's normal newest-wins
    behavior instead of guessing."""
    if old.operator != new.operator or old.operator not in ("gte", "lte"):
        return new
    try:
        old_value, new_value = float(old.value), float(new.value)
    except (TypeError, ValueError):
        return new
    if old.operator == "gte":
        return new if new_value >= old_value else old
    return new if new_value <= old_value else old


def merge_filters(existing: list[Filter], incoming: list[Filter]) -> list[Filter]:
    """Incoming filters replace existing ones with the same key (field+skill).
    This is what makes 'actually, Bangalore instead' update rather than
    duplicate the location filter. Exception: two years-based experience
    filters on the SAME key (see _EXPERIENCE_YEAR_FIELDS) don't just take
    whichever is newest -- they resolve to whichever demands MORE experience,
    see _stronger_experience_filter's docstring for the real bug this fixes."""
    merged: dict[tuple, Filter] = {f.key(): f for f in existing}
    for f in incoming:
        key = f.key()
        prior = merged.get(key)
        if prior is not None and f.field in _EXPERIENCE_YEAR_FIELDS:
            merged[key] = _stronger_experience_filter(prior, f)
        else:
            merged[key] = f
    return list(merged.values())


def merge_alternative_groups(
    existing: list[AlternativeGroup],
    incoming: list[AlternativeGroup],
    replace_all: bool,
) -> list[AlternativeGroup]:
    """Wholesale replace, NOT a per-key merge like merge_filters -- an
    "either A-route or B-route" statement is one coherent clause; there is
    no sensible interpretation of merging leaf filters from an OLDER route
    with a NEWER, textually different one. If the new turn states
    alternative_groups at all (non-empty `incoming`), they fully replace
    whatever was active. If the new turn states NONE and this isn't a
    replace_all turn, the existing groups carry over unchanged -- same
    continuity convention as an unmentioned flat filter surviving a
    refinement query. Under replace_all=True, `incoming` wins outright even
    if empty (mirrors how replace_all already fully replaces `filters`)."""
    if incoming or replace_all:
        return incoming
    return existing


def _chip_label_text(f: Filter) -> str:
    icon = _FIELD_ICON.get(f.field, "🔖")
    if f.field in ("skill_experience", "domain_experience"):
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


def chip_label(f: Filter) -> str:
    """Per-field label text, plus a "~" prefix for a soft preference
    (Filter.hard=False, see schema_v2's "nice to have") -- a plain-text
    fallback for anything reading just the label string, e.g. console/API
    testing. The structural `Chip.hard` flag (see to_chips) is what the
    real UI should actually key off of for styling (dashed border,
    different color, etc.), not this text convention."""
    text = _chip_label_text(f)
    return f"~ {text}" if not f.hard else text


def to_chips(
    filters: list[Filter], alternative_groups: list[AlternativeGroup] | None = None,
) -> list[Chip]:
    """One chip per flat filter, plus (if `alternative_groups` is non-empty)
    exactly ONE additional synthetic chip representing the whole OR-of-
    routes requirement -- never one chip per group, which would misrepresent
    "any ONE of these routes is required" as a set of unrelated separate
    facts. Uses the synthetic field marker "_alternative_group" (not a real
    ALLOWED_FIELDS value) so the UI can style it distinctly from an ordinary
    fact pill -- see search-ui/index.html's renderChips."""
    chips = [
        Chip(label=chip_label(f), field=f.field, skill=f.skill, hard=f.hard)
        for f in filters
    ]
    if alternative_groups:
        route_texts = [
            " AND ".join(_chip_label_text(f) for f in g.filters)
            for g in alternative_groups if g.filters
        ]
        if route_texts:
            chips.append(Chip(
                label=" OR ".join(f"({t})" for t in route_texts),
                field="_alternative_group", hard=True,
            ))
    return chips
