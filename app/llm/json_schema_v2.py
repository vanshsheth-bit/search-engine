"""JSON schema for PROMPT_SCHEMA=v2 (the extraction-first design).

Hybrid, by design: schema_v2's `structured`/`tools`/`domain_hint` buckets
(the actual innovation -- the LLM reports raw spans, never a resolved final
value) for FILTER_CANDIDATES, grafted onto the SAME intent-management
scaffolding `json_schema.py` already has for CLARIFY/LOOKUP/
UNSUPPORTED_FILTER/EXPERIENCE_SEARCH -- those four intents have nothing to
do with the extraction-vs-resolution split this migration is about, and
schema_v2 as originally drafted didn't cover them at all (no `logic`, no
`replace_all`, no `clarify_*`/`options`/`candidate_ref`/`lookup_field`/
`experience_query`, and its `intent` enum omitted EXPERIENCE_SEARCH
entirely -- confirmed by reading it directly, not assumed).

Field families for `structured` are DERIVED from vocabulary.py's
FIELD_TYPES/OPERATORS_BY_TYPE rather than hand-copied, so this schema can
never drift out of sync with the single source of truth every other part
of the system already uses (same principle as json_schema.py's own
docstring). This also fixes a real bug in schema_v2 as drafted: its STRING
branch fixed `value` to a bare string even though its own operator set
included "in"/"not_in" (which need an array) -- confirmed with
jsonschema.validate. Here, `value` is `anyOf: [T, array-of-T]` for any
family whose operator set includes the list operators, derived the same
way the family itself is.
"""
from __future__ import annotations

from collections import defaultdict

from app.core.vocabulary import (
    FIELD_TYPES,
    LIST_OPERATORS,
    OPERATORS_BY_TYPE,
    VALID_INTENTS,
)
from app.llm.filter_item_schema import alternative_groups_schema

# schema_v2's own broad-category list for domain_hint -- deliberately NOT
# the real 212-category subdomain taxonomy (see LLMOutput.domain_hint's
# docstring: this is a coarse hint, not resolved into a Filter yet, so it
# doesn't need to line up with real candidate data the way a real filter
# value would).
DOMAINS = [
    "Engineering", "Software Engineering", "ML/AI", "Design/Creative",
    "Finance", "HR / People", "Healthcare", "Management / Leadership",
    "Legal", "Marketing", "Media / Journalism", "Operations",
    "Presales/Solutions", "Sales",
]

# `skill`/`skill_experience` are excluded from the generic per-type grouping
# below: `skill` goes through the dedicated `tools` bucket (with its own
# exact/expand semantics `structured` items have no room for), and
# `skill_experience` needs an extra `skill` key no other field has, so it
# gets its own oneOf branch instead of being folded into the generic
# "number" family.
_SKILL_FIELDS = {"skill", "skill_experience"}

_JSON_TYPE = {"string": "string", "number": "number", "boolean": "boolean"}


def _value_schema(value_type: str, ops: set[str]) -> dict:
    base = {"type": _JSON_TYPE.get(value_type, "string")}
    if ops & LIST_OPERATORS:
        return {"anyOf": [base, {"type": "array", "items": base}]}
    return base


def _fields_by_type() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for field, ftype in FIELD_TYPES.items():
        if field in _SKILL_FIELDS:
            continue
        grouped[ftype].append(field)
    return grouped


def _family_branch(fields: list[str], value_type: str, ops: set[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "field": {"type": "string", "enum": sorted(fields)},
            "operator": {"type": "string", "enum": sorted(ops)},
            "raw_text": {"type": "string"},
            "value": _value_schema(value_type, ops),
            "hard": {"type": "boolean"},
        },
        "required": ["field", "operator", "value", "hard"],
        # Confirmed live failure this prevents: without this, nothing stops
        # the model from bolting an unrelated key onto the wrong branch --
        # e.g. a "domain" item carrying a stray "skill" property copied
        # from context, producing a validly-shaped-but-semantically-
        # nonsensical filter instead of Ollama's grammar rejecting it.
        "additionalProperties": False,
    }


def _skill_experience_branch() -> dict:
    ops = OPERATORS_BY_TYPE["number"]
    return {
        "type": "object",
        "properties": {
            "field": {"const": "skill_experience"},
            "operator": {"type": "string", "enum": sorted(ops)},
            "raw_text": {"type": "string"},
            "value": {"type": "number"},
            "skill": {"type": "string"},
            "hard": {"type": "boolean"},
        },
        "required": ["field", "operator", "value", "skill", "hard"],
        "additionalProperties": False,
    }


def _structured_item_schema() -> dict:
    """oneOf over field families (plus skill_experience's own branch) --
    the grammar makes an invalid field+operator+value-type combination
    (e.g. an ordinal field with a "contains" operator, or a number field
    with a string value) structurally undecodable, not just caught
    downstream by validation.py. Proven with jsonschema.validate against
    both this schema and the current flat one (see the migration plan's
    G-numbered corrections)."""
    branches = [_skill_experience_branch()]
    for ftype, fields in _fields_by_type().items():
        branches.append(_family_branch(fields, ftype, OPERATORS_BY_TYPE[ftype]))
    return {"oneOf": branches}


def _tool_item_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "raw_text": {"type": "string"},
            "match_mode": {"type": "string", "enum": ["exact", "expand"]},
            "hard": {"type": "boolean"},
        },
        "required": ["raw_text", "match_mode", "hard"],
        "additionalProperties": False,
    }


def build_filter_json_schema_v2() -> dict:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": sorted(VALID_INTENTS)},
            "logic": {"type": "string", "enum": ["AND", "OR", "NOT"]},
            "replace_all": {"type": "boolean"},
            "structured": {"type": "array", "items": _structured_item_schema()},
            "tools": {"type": "array", "items": _tool_item_schema()},
            "domain_hint": {"type": "array", "items": {"type": "string", "enum": DOMAINS}},
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            # AlternativeGroup members are ALWAYS v1-shaped/already-resolved
            # (field/operator/value directly, not a raw span for `structured`
            # to later resolve) -- see AlternativeGroup's docstring in
            # app/models/schemas.py for why: every real "either eligibility
            # route A or route B" pattern found only needs ordinal/numeric
            # leaves that never need taxonomy skill-resolution.
            "alternative_groups": alternative_groups_schema(),
            # Carried over unchanged from json_schema.py -- CLARIFY/LOOKUP/
            # EXPERIENCE_SEARCH/UNSUPPORTED_FILTER have nothing to do with
            # the extraction-vs-resolution split; these fields are resolved
            # exactly as they are today regardless of which schema produced
            # them (service.py's handling of them is unchanged).
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "clarify_field": {"type": "string", "enum": sorted(FIELD_TYPES)},
            "clarify_skill": {"type": "string"},
            "clarify_operator": {"type": "string",
                                 "enum": sorted(OPERATORS_BY_TYPE["number"] & {"gte", "lte", "gt", "lt"})},
            "clarify_value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
            "clarify_unit": {"type": "string"},
            "message": {"type": "string"},
            "candidate_ref": {"type": "string"},
            "lookup_field": {"type": "string", "enum": sorted(FIELD_TYPES)},
            "experience_query": {"type": "string"},
        },
        "required": ["intent", "replace_all"],
        "additionalProperties": False,
    }
