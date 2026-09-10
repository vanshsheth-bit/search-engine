"""Builds the JSON schema handed to Ollama's `format` param for constrained
decoding, derived from the vocabulary so it never drifts."""
from __future__ import annotations

from app.core.vocabulary import (
    ALLOWED_FIELDS,
    NUMERIC_OPERATORS,
    VALID_INTENTS,
)
from app.llm.filter_item_schema import alternative_groups_schema, resolved_filter_item_schema


def build_filter_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": sorted(VALID_INTENTS)},
            "logic": {"type": "string", "enum": ["AND", "OR", "NOT"]},
            "replace_all": {"type": "boolean"},
            "filters": {
                "type": "array",
                "items": resolved_filter_item_schema(),
            },
            "alternative_groups": alternative_groups_schema(),
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
