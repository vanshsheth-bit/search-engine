"""Service layer: orchestrates the full pipeline.

query -> LLM translate -> merge with session -> validate -> engine -> response
"""
from __future__ import annotations

import logging
import re

from app.core.candidates import get_available_fields, get_matched_candidates
from app.core.engine import apply_spec
from app.core.lookup import answer_lookup, resolve_candidate
from app.core.merge import merge_filters, to_chips
from app.core.session import SessionStore, default_store
from app.core.skill_taxonomy import expand_skill_filters
from app.core.validation import validate_filters
from app.llm.client import LLMClient
from app.models.schemas import (
    Filter,
    FilterResponse,
    FilterSpec,
    LLMOutput,
    PatchStateRequest,
    PendingClarify,
    SessionState,
)

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _extract_number(text: str) -> float | int | None:
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    n = float(m.group(1))
    return int(n) if n.is_integer() else n


def _extract_unit(text: str) -> str | None:
    low = text.lower()
    if "month" in low:
        return "months"
    if "day" in low:
        return "days"
    if "year" in low:
        return "years"
    return None


class FilterService:
    def __init__(
        self,
        llm: LLMClient | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.store = store or default_store

    # ------------------------------------------------------------------ #
    # Main NL entry point
    # ------------------------------------------------------------------ #
    def filter_by_query(
        self, query: str, job_id: str, session_id: str, reset: bool = False
    ) -> FilterResponse:
        if reset:
            self.store.clear(session_id, job_id)

        current = self.store.get(session_id, job_id)
        spec = current.spec

        # A pending lookup ("which candidate did you mean?") takes priority
        # over the LLM entirely. A bare name typed/clicked in reply has no
        # other sensible reading as a fresh query, and the LLM has no way to
        # recover which field was even being asked about from a bare name
        # alone -- so resolve it directly against who was actually offered,
        # deterministically, same as everywhere else in this system.
        if current.pending_lookup_field:
            resolution = resolve_candidate(query, current.last_candidates)
            if resolution.candidate is not None:
                answer = answer_lookup(resolution.candidate, current.pending_lookup_field)
                self.store.set(session_id, job_id, SessionState(
                    spec=spec, last_candidates=current.last_candidates,
                ))
                return FilterResponse(
                    status="answer", logic=spec.logic,
                    filters=spec.filters, chips=to_chips(spec.filters),
                    message=answer,
                )
            # Didn't match one of the offered names -- treat as abandoning
            # the pending lookup and fall through to a fresh query below.

        # Same idea for a pending CLARIFY ("what minimum years of
        # experience?"): a short reply ("2+ years", or clicking that exact
        # option) has no reliable interpretation once sent to the LLM with
        # no memory of the question -- a bare fragment like "2+ years" isn't
        # enough context for the model to know what field it answers. Extract
        # the number deterministically instead. Only for SHORT replies --
        # a longer reply might be a genuinely new compound query (e.g. "at
        # least 5 years, also based in Delhi"), which this simple extraction
        # would wrongly reduce to just the number and silently drop the
        # rest of -- let that fall through to the LLM instead.
        if current.pending_clarify and len(query.split()) <= 6:
            value = _extract_number(query)
            if value is not None:
                pc = current.pending_clarify
                filt = Filter(field=pc.field, operator=pc.operator, value=value, skill=pc.skill)
                if pc.field == "notice_period":
                    filt.unit = _extract_unit(query) or "days"
                merged = merge_filters(spec.filters, [filt])
                return self._validate_apply_persist(merged, spec.logic, job_id, session_id)
            # No number in the reply -- treat as abandoning the pending
            # clarification and fall through to a fresh query below.

        llm_out: LLMOutput = self.llm.translate(
            query, [f.model_dump(exclude_none=True) for f in spec.filters]
        )

        if llm_out.intent == "CLARIFY":
            pending = None
            if llm_out.clarify_field:
                pending = PendingClarify(
                    field=llm_out.clarify_field,
                    operator=llm_out.clarify_operator or "gte",
                    skill=llm_out.clarify_skill,
                )
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                pending_clarify=pending,
            ))
            return FilterResponse(
                status="clarify",
                question=llm_out.question or "Could you clarify your filter?",
                options=llm_out.options,
                logic=spec.logic,
                filters=spec.filters,
                chips=to_chips(spec.filters),
            )

        if llm_out.intent == "UNSUPPORTED_FILTER":
            return FilterResponse(
                status="unsupported",
                message=llm_out.message or "That filter is not supported.",
                logic=spec.logic,
                filters=spec.filters,
                chips=to_chips(spec.filters),
            )

        if llm_out.intent == "LOOKUP":
            return self._answer_lookup(llm_out, current, spec, job_id, session_id)

        # FILTER_CANDIDATES. Expand skill concepts ("machine learning" ->
        # its real tools) against the curated taxonomy before merging into
        # session state, so what gets stored/matched is already the precise
        # expansion, not the bare concept term.
        expanded_filters = expand_skill_filters(llm_out.filters)
        merged = merge_filters(spec.filters, expanded_filters)
        return self._validate_apply_persist(
            merged, llm_out.logic, job_id, session_id,
            extra_message=llm_out.message,
        )

    # ------------------------------------------------------------------ #
    # LOOKUP: a question about one already-shown candidate, not a new
    # filter. Resolved entirely deterministically against real stored data
    # -- the LLM only identified WHICH question, never the answer itself.
    # ------------------------------------------------------------------ #
    def _answer_lookup(
        self, llm_out: LLMOutput, current: SessionState, spec: FilterSpec,
        job_id: str, session_id: str,
    ) -> FilterResponse:
        base = dict(
            status="unsupported", logic=spec.logic,
            filters=spec.filters, chips=to_chips(spec.filters),
        )

        if not llm_out.lookup_field:
            return FilterResponse(
                **base,
                message="I wasn't sure what you were asking about that "
                        "candidate -- could you rephrase?",
            )

        resolution = resolve_candidate(llm_out.candidate_ref, current.last_candidates)

        if resolution.candidate is None and not resolution.ambiguous_names:
            return FilterResponse(
                **base,
                message="I don't have a candidate in view to answer that about "
                        "-- search for someone first.",
            )

        if resolution.candidate is None:
            # Remember what was being asked, so the next message (a bare
            # name, typed or clicked) can complete it directly.
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                pending_lookup_field=llm_out.lookup_field,
            ))
            names = ", ".join(resolution.ambiguous_names)
            return FilterResponse(
                status="clarify",
                logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
                question=f"Which candidate did you mean -- {names}?",
                options=resolution.ambiguous_names,
            )

        answer = answer_lookup(resolution.candidate, llm_out.lookup_field)
        return FilterResponse(
            status="answer",
            logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
            message=answer,
        )

    # ------------------------------------------------------------------ #
    # Deterministic chip edit (no LLM)
    # ------------------------------------------------------------------ #
    def patch_state(self, req: PatchStateRequest) -> FilterResponse:
        return self._validate_apply_persist(
            req.filters, req.logic, req.job_id, req.session_id
        )

    def clear(self, session_id: str, job_id: str) -> FilterResponse:
        candidates = get_matched_candidates(job_id)
        # Clearing filters still shows the full pool -- remember it too, so
        # "which college did he go to" keeps working right after a reset.
        self.store.set(session_id, job_id, SessionState(last_candidates=candidates))
        return FilterResponse(
            status="ok",
            total=len(candidates),
            showing=len(candidates),
            candidates=candidates,
        )

    # ------------------------------------------------------------------ #
    # Shared tail: validate -> apply -> persist -> respond
    # ------------------------------------------------------------------ #
    def _validate_apply_persist(
        self, filters: list[Filter], logic: str, job_id: str, session_id: str,
        extra_message: str | None = None,
    ) -> FilterResponse:
        available = get_available_fields(job_id)
        result = validate_filters(filters, available)

        if not result.ok:
            status = "unsupported" if result.unsupported else "error"
            # Persist only the filters that were valid before the bad one?
            # Safer: do not mutate stored state on invalid input.
            current = self.store.get(session_id, job_id)
            return FilterResponse(
                status=status,
                message=result.error,
                logic=current.spec.logic,
                filters=current.spec.filters,
                chips=to_chips(current.spec.filters),
            )

        spec = FilterSpec(logic=logic, filters=result.filters)
        candidates = get_matched_candidates(job_id)
        filtered = apply_spec(candidates, spec)

        # Persist the new valid state, including who's now in view -- this
        # is what a later LOOKUP ("which college did he go to") resolves
        # against.
        self.store.set(session_id, job_id, SessionState(spec=spec, last_candidates=filtered))

        # One or more filters in this request were dropped (unsupported field,
        # bad value, etc.) but at least one other filter was still valid --
        # apply what's real and say what got skipped, instead of failing the
        # whole request over one bad clause (see validate_filters).
        skip_note = (
            "Couldn't apply: " + "; ".join(result.skipped) + "."
            if result.skipped else None
        )
        # extra_message: the LLM's own note about a concept it recognized as
        # unsupported but has no ALLOWED_FIELDS equivalent to even express as
        # a droppable Filter (e.g. "product-based vs service-based") -- see
        # prompt.py's compound-query rule. Merged with skip_note so both
        # sources of "here's what couldn't be applied" reach the recruiter.
        skip_note = " ".join(m for m in (extra_message, skip_note) if m) or None

        if not filtered:
            return FilterResponse(
                status="no_match",
                total=len(candidates),
                showing=0,
                logic=spec.logic,
                filters=spec.filters,
                chips=to_chips(spec.filters),
                message=skip_note or "No candidates match these filters.",
                suggestions=_no_match_suggestions(spec.filters),
            )

        return FilterResponse(
            status="ok",
            total=len(candidates),
            showing=len(filtered),
            logic=spec.logic,
            filters=spec.filters,
            chips=to_chips(spec.filters),
            candidates=filtered,
            message=skip_note,
        )


def _no_match_suggestions(filters: list[Filter]) -> list[str]:
    tips = ["Remove one of the filters", "Search all locations"]
    for f in filters:
        if f.field == "skill_experience":
            tips.append(f"Reduce the {f.skill} experience requirement")
        if f.field == "experience":
            tips.append("Lower the minimum experience")
    # de-dupe, keep order
    seen, out = set(), []
    for t in tips:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
