"""Converts the v2 schema's raw JSON dict into an `LLMOutput`.

This is a real, necessary step -- there is no existing code that turns
`{"structured": [...], "tools": [...], "ambiguities": [...]}` into an
`LLMOutput` object; `LLMClient.translate()` today does one flat
`LLMOutput.model_validate(data)` call, which works for v1's shape but
would leave v2's `ambiguities` silently unread (Pydantic's default
`extra='ignore'` drops unknown keys with no error) if the raw dict were
handed to `model_validate` directly.

`ambiguities` needs an explicit home: schema_v2's own model produces this
list for anything it noticed but couldn't confidently resolve, separate
from `question` (which is only set for an already-decided CLARIFY). This
adapter decides where that goes, mirroring exactly how service.py already
surfaces "understood but not resolved" information for the v1 path:
- CLARIFY with no `question` but real `ambiguities` -> synthesize
  `question` from them (mirrors service.py's own fallback,
  `llm_out.question or "Could you clarify your filter?"` -- a generic
  fallback would silently swallow the real ambiguity text).
- Anything else (typically FILTER_CANDIDATES with real filters ALSO
  present) -> folded into `message`, which service.py already threads
  through as `extra_message` into `_validate_apply_persist` for the v1
  path's compound-unsupported-concept case (e.g. "product-based vs
  service-based" with no ALLOWED_FIELDS equivalent) -- so it surfaces next
  to real results instead of vanishing.
"""
from __future__ import annotations

from app.models.schemas import AlternativeGroup, Filter, LLMOutput, StructuredItem, ToolItem


def parse_v2_output(raw: dict) -> LLMOutput:
    ambiguities = raw.get("ambiguities") or []

    llm_out = LLMOutput(
        intent=raw.get("intent", "UNSUPPORTED_FILTER"),
        logic=raw.get("logic", "AND"),
        replace_all=raw.get("replace_all", False),
        structured=[StructuredItem(**item) for item in raw.get("structured") or []],
        tools=[ToolItem(**item) for item in raw.get("tools") or []],
        domain_hint=raw.get("domain_hint") or [],
        # Group members are ALWAYS v1-shaped/already-resolved (field/
        # operator/value directly) regardless of the v2 schema's usual
        # extraction-first shape -- see AlternativeGroup's docstring.
        alternative_groups=[
            AlternativeGroup(filters=[Filter(**f) for f in group.get("filters") or []])
            for group in raw.get("alternative_groups") or []
        ],
        question=raw.get("question"),
        options=raw.get("options") or [],
        clarify_field=raw.get("clarify_field"),
        clarify_skill=raw.get("clarify_skill"),
        clarify_operator=raw.get("clarify_operator"),
        clarify_value=raw.get("clarify_value"),
        clarify_unit=raw.get("clarify_unit"),
        message=raw.get("message"),
        candidate_ref=raw.get("candidate_ref"),
        lookup_field=raw.get("lookup_field"),
        experience_query=raw.get("experience_query"),
    )

    if not ambiguities:
        return llm_out

    if llm_out.intent == "CLARIFY" and not llm_out.question:
        return llm_out.model_copy(update={"question": "; ".join(ambiguities)})

    joined = "; ".join(ambiguities)
    combined_message = f"{llm_out.message} {joined}" if llm_out.message else joined
    return llm_out.model_copy(update={"message": combined_message})
