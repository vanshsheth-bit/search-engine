"""Shared "resolved filter" item shape (field/operator/value/skill/unit) --
the same shape a v1 filter has always used. Extracted so json_schema.py and
json_schema_v2.py both build their `alternative_groups` array from the
identical definition instead of hand-duplicating it and risking drift --
AlternativeGroup members are always v1-shaped/already-resolved regardless of
which top-level schema (v1 or v2) is active, see AlternativeGroup's
docstring in app/models/schemas.py."""
from __future__ import annotations

from app.core.vocabulary import ALLOWED_FIELDS, ALLOWED_OPERATORS


def resolved_filter_item_schema() -> dict:
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


def alternative_groups_schema() -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "filters": {"type": "array", "items": resolved_filter_item_schema()},
            },
            "required": ["filters"],
        },
    }
