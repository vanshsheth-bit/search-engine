"""Builds the JSON schema handed to Ollama's `format` param for constrained
decoding, derived from the vocabulary so it never drifts."""
from __future__ import annotations

from app.core.vocabulary import (
    ALLOWED_FIELDS,
    ALLOWED_OPERATORS,
    NUMERIC_OPERATORS,
    VALID_INTENTS,
)


def _filter_item_schema() -> dict:
    """Shape of ONE filter -- reused for `filters`, `preferred_filters`, and
    each branch inside `alternative_groups` so all three stay in sync with
    ALLOWED_FIELDS/ALLOWED_OPERATORS automatically."""
    return {
        "type": "object",
        "properties": {
            "field": {"type": "string", "enum": ALLOWED_FIELDS},
            "operator": {"type": "string", "enum": ALLOWED_OPERATORS},
            "skill": {"type": "string"},
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "array"},
                ]
            },
            "unit": {"type": "string"},
        },
        "required": ["field", "operator", "value"],
    }


def build_filter_json_schema() -> dict:
    filter_item = _filter_item_schema()
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": sorted(VALID_INTENTS)},
            "logic": {"type": "string", "enum": ["AND", "OR", "NOT"]},
            "replace_all": {"type": "boolean"},
            "filters": {"type": "array", "items": filter_item},
            # See AlternativeGroup's docstring in schemas.py -- a bounded,
            # non-recursive "either this WHOLE requirement-set or that one"
            # spanning DIFFERENT fields. NOT for a single field's own list
            # of alternative values (use `filters` with operator "in" for
            # that instead -- see prompt.py rule 3c).
            "alternative_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "branches": {
                            "type": "array",
                            "items": {"type": "array", "items": filter_item},
                        },
                    },
                    "required": ["branches"],
                },
            },
            # See LLMOutput.preferred_filters -- soft-preference language
            # ("prefer", "ideally", "bonus if"), never excludes a candidate.
            "preferred_filters": {"type": "array", "items": filter_item},
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "clarify_field": {"type": "string", "enum": ALLOWED_FIELDS},
            "clarify_skill": {"type": "string"},
            "clarify_operator": {"type": "string", "enum": sorted(NUMERIC_OPERATORS)},
            "clarify_value": {"anyOf": [{"type": "string"}, {"type": "number"}]},
            "clarify_unit": {"type": "string"},
            "message": {"type": "string"},
            "candidate_ref": {"type": "string"},
            "lookup_field": {"type": "string", "enum": ALLOWED_FIELDS},
            "experience_query": {"type": "string"},
        },
        "required": ["intent", "replace_all"],
    }
