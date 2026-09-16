"""Resolves the v2 extraction schema's raw spans into real, engine-ready
Filter objects.

DESIGN RULE (matches schema_v2's own docstring): the LLM reports what the
recruiter SAID (a raw span, a coarse bucket, hard-vs-soft, exact-vs-expand);
this module decides what it MEANS. It is the only place that turns a
`StructuredItem`/`ToolItem` into a `Filter` -- validation.py, engine.py, and
merge.py never see the v2 shapes directly, only the `Filter` objects this
produces, so the entire downstream pipeline (validate -> apply_spec ->
fuzzy skill widening -> soft-preference ranking -> merge/session) is
completely unaware of which schema version produced its input.

Pure function, no I/O beyond skill_taxonomy.py's already-cached lookups:
no validation, no session state, no candidate data. `service.py` decides
what to do with the results (country canonicalization, merging with
session state, validation) exactly the same way it already does for the v1
`expand_skill_filters()` path -- this module's only job is the translation.
"""
from __future__ import annotations

import re

from app.core.skill_taxonomy import canonicalize, expand_skill_term
from app.core.vocabulary import FIELD_TYPES
from app.models.schemas import Filter, LLMOutput, StructuredItem, ToolItem

# Fields whose value is safe to ground directly against the query text as a
# fallback when raw_text is missing (see _grounding_terms) -- "string" and
# "ordinal" values are close enough to what a recruiter actually typed
# ("High", "Master", "Product") that a literal substring check is
# meaningful. Deliberately EXCLUDES "number" and "boolean": a numeric value
# is often derived/paraphrased rather than a literal digit in the query
# ("five years" -> 5, a unit-converted notice_period), and a boolean value
# ("True"/"False") never appears in natural-language text at all -- for
# both, a literal-substring check would reject perfectly legitimate filters,
# not catch hallucinated ones.
_GROUNDABLE_VALUE_FIELD_TYPES = {"string", "ordinal"}

# Real, reported live bug found via a 20-case edge/mechanism sweep:
# college_tier/company_tier ARE "ordinal" (see _GROUNDABLE_VALUE_FIELD_TYPES
# above) but their canonical values are a fixed Low/Medium/High scale that a
# recruiter essentially never types literally -- they say "top-tier
# university", "Tier 1 companies", "premier", "reputed", never the word
# "High" itself. When the model (wrongly) omits raw_text for one of these
# two fields, the value-fallback grounding check below was comparing
# "High"/"Low" against query text that could never contain it by
# construction, silently dropping a completely legitimate filter as if it
# were the SAME kind of hallucination the fallback was built to catch (a
# company_type value with zero relation to garbled query text) -- confirmed
# live: "a top-tier university, not Infosys or TCS" and "PhD from a Tier 1
# university" both lost the tier requirement entirely this way. `education`
# stays in the general ordinal fallback (its own canonical values --
# "Bachelor's", "Master's", "PhD" -- DO appear at or near verbatim in real
# recruiter text), only these two are excluded.
_TIER_VALUE_NEVER_LITERAL_FIELDS = {"college_tier", "company_tier"}


def _mentioned_in_query(term: str, query: str) -> bool:
    """Case-insensitive check that `term` actually appears in `query`.
    Word-boundary matched for short terms (<=3 characters, e.g. "ML", "R",
    "Go") so a coincidental substring inside an unrelated word ("html",
    "sample") can't silently defeat the check; plain substring for longer,
    naturally distinctive terms, where that risk is negligible."""
    if not term:
        return True
    if len(term) <= 3:
        return re.search(rf"\b{re.escape(term)}\b", query, re.IGNORECASE) is not None
    return term.lower() in query.lower()


def _grounding_terms(item: StructuredItem) -> list[str]:
    """Every string worth checking `item` against the query text for.
    `raw_text` is the primary, expected signal (v2's own design contract:
    it's supposed to always be a literal span of what was said); `.skill`
    is the established fallback for skill_experience/domain_experience,
    whose raw_text may legitimately be absent (see resolve_filters's
    docstring).

    For every OTHER field, when the model omits raw_text too, there was
    previously NO fallback at all -- the item passed through completely
    unchecked. Real, reported live bug: a `company_type` item with
    raw_text=null and value=["Product","Both"] had ZERO connection to the
    actual query text ("candidate worked in high tire company", a garbled
    "high tier company" typo) and sailed through anyway. Falls back to the
    item's own value(s) here instead -- but ONLY for string/ordinal-typed
    fields (see _GROUNDABLE_VALUE_FIELD_TYPES): a numeric value is often
    derived/paraphrased ("five years" -> 5) and a boolean value never
    appears in natural-language text at all, so checking either literally
    would reject legitimate filters, not catch hallucinated ones.

    Returns [] when there is nothing safe to check (numeric/boolean field,
    empty value) -- callers must treat that as "nothing to ground on",
    passing the item through unchanged, not as "grounding failed"."""
    if item.raw_text:
        return [item.raw_text]
    if item.field in ("skill_experience", "domain_experience") and item.skill:
        return [item.skill]
    if item.field in _TIER_VALUE_NEVER_LITERAL_FIELDS:
        return []
    if FIELD_TYPES.get(item.field) not in _GROUNDABLE_VALUE_FIELD_TYPES:
        return []
    values = item.value if isinstance(item.value, list) else [item.value]
    return [str(v) for v in values if v not in (None, "")]


def active_filter_terms(filters: list[Filter]) -> frozenset[str]:
    """Every value string (and skill_experience/domain_experience's
    `.skill`) across `filters`, lowercased -- the "already active,
    regardless of this turn's own wording" side of resolve_filters'
    active_terms parameter. See that parameter's docstring for the bug
    this exists to fix: distinguishing a genuine context-bleed
    hallucination from a harmless redundant restatement of something the
    recruiter already has active."""
    terms: set[str] = set()
    for f in filters:
        values = f.value if isinstance(f.value, list) else [f.value]
        terms.update(str(v).lower() for v in values if v not in (None, ""))
        if f.skill:
            terms.add(f.skill.lower())
    return frozenset(terms)


def _structured_to_filter(item: StructuredItem) -> Filter:
    return Filter(
        field=item.field,
        operator=item.operator,
        value=item.value,
        skill=item.skill,
        hard=item.hard,
    )


def _tool_to_filter(item: ToolItem) -> Filter:
    """"exact" -> canonicalize only (alias fix, never widened -- a specific
    named tool was demanded, and widening it to related-but-different tools
    would match someone who does NOT have what was actually asked for).
    "expand" -> expand_skill_term(), which widens to related tools when the
    taxonomy has something to say (returns a list) and falls back to
    canonicalize()-only when it doesn't (returns None) -- the same
    three-way signal (found-with-relations / found-no-relations / not-found)
    expand_skill_term already gives the v1 path via expand_skill_filters()."""
    if item.match_mode == "expand":
        expanded = expand_skill_term(item.raw_text)
        if expanded:
            return Filter(field="skill", operator="in", value=expanded, hard=item.hard)
    return Filter(field="skill", operator="contains",
                  value=canonicalize(item.raw_text), hard=item.hard)


def resolve_filters(
    llm_out: LLMOutput, query: str, active_terms: frozenset[str] = frozenset(),
) -> tuple[list[Filter], list[str]]:
    """v2 LLMOutput (structured/tools/domain_hint populated, filters empty)
    -> (filters, skip_notes). skip_notes surfaces things that were
    understood but deliberately not turned into a filter -- e.g.
    domain_hint (see LLMOutput.domain_hint's docstring for why: it's a
    coarse 14-category hint, the real candidate `domain` data is 212
    fine-grained subdomains that often share no substring with the hint at
    all, so shipping it as a silently-inert soft filter would look
    tested-and-working on coincidental word overlaps while doing nothing
    everywhere else) -- callers should fold skip_notes into the same
    extra_message mechanism _validate_apply_persist already uses for the
    v1 path's compound-unsupported-concept case, not discard them.

    `query` grounds each item against what the recruiter actually typed --
    real, reported live bug: asking "who has worked on supply chain
    platform" (a session's SECOND turn, after an earlier unrelated "give me
    a ML guy") produced a fabricated
    {"field":"skill_experience","skill":"ML","value":1} structured item --
    zero mention of ML anywhere in this turn's query, apparently bled in
    from the earlier turn despite this turn's OWN `replace_all: true`
    asserting it's a standalone reinterpretation. v2's whole design
    principle is that raw_text/skill is a literal SPAN of what was actually
    said (see schema_v2's own docstring) -- a term that doesn't appear in
    the query AT ALL violates that by construction, so it's dropped here
    rather than trusted.

    `active_terms` (lowercased, see active_filter_terms) is what's already
    active in the SESSION, independent of this turn's own wording -- it
    exists to tell that genuine hallucination apart from a second, real,
    reported live bug: a follow-up that ADDS to an existing filter ("also
    needs AWS experience", after an earlier turn set Python) caused the
    model to REDUNDANTLY restate "Python" in its own raw output for this
    turn -- ordinary, expected multi-turn behavior (the recruiter's "also"
    implicitly refers to what's already active), not a hallucination. Since
    "python" isn't literally in THIS turn's text, the grounding check above
    drops it exactly like the genuine ML case -- correctly, since
    merge_filters (in service.py, using spec.filters -- the SAME already-
    active state active_terms is built from) will re-add it from session
    state regardless, so nothing is actually lost -- but saying so with
    "...carried over from an earlier, UNRELATED question" was confusing and
    FACTUALLY WRONG: Python is not unrelated, it is the exact thing this
    turn is building on. When the dropped term is already in
    `active_terms`, this is a complete non-event and stays silent; only a
    term connected to NEITHER this turn's text NOR anything already active
    gets the explanatory note, which is the only case it was ever meant
    for."""
    filters = []
    skip_notes = []
    for item in llm_out.structured:
        terms = _grounding_terms(item)
        if terms and not any(_mentioned_in_query(t, query) for t in terms):
            if not any(t.lower() in active_terms for t in terms):
                skip_notes.append(
                    f'Ignored a mention of "{terms[0]}" that doesn\'t appear in this '
                    f"message -- it looks like it carried over from an earlier, "
                    f"unrelated question."
                )
            continue
        filters.append(_structured_to_filter(item))
    for item in llm_out.tools:
        if not _mentioned_in_query(item.raw_text, query):
            if item.raw_text.lower() not in active_terms:
                skip_notes.append(
                    f'Ignored a mention of "{item.raw_text}" that doesn\'t appear '
                    f"in this message -- it looks like it carried over from an "
                    f"earlier, unrelated question."
                )
            continue
        filters.append(_tool_to_filter(item))

    if llm_out.domain_hint:
        skip_notes.append(
            "Noted the likely domain (" + ", ".join(llm_out.domain_hint) + ") "
            "but didn't filter on it -- ask for a specific skill, title, or "
            "company instead for a precise result."
        )
    return filters, skip_notes
