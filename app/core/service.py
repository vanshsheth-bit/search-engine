"""Service layer: orchestrates the full pipeline.

query -> LLM translate -> merge with session -> validate -> engine -> response
"""
from __future__ import annotations

import logging
import re

from app.core.candidates import canonicalize_country, get_available_fields, get_matched_candidates
from app.core.engine import apply_spec, matches_filter
from app.core import experience_index
from app.core.lookup import answer_lookup, resolve_candidate
from app.core.merge import merge_filters, to_chips
from app.core.session import SessionStore, default_store
from app.core.skill_taxonomy import (
    canonicalize, expand_skill_filters, related_terms_for, skill_names_of,
)
from app.core.semantic import MIN_SIMILARITY, term_similarities
from app.core.validation import validate_filters
from app.llm.client import LLMClient
from app.llm.skill_verify import verify_skill_candidates
from app.models.schemas import (
    ChatTurn,
    Chip,
    Filter,
    FilterResponse,
    FilterSpec,
    LLMOutput,
    PatchStateRequest,
    PendingClarify,
    PendingCombine,
    SessionState,
)

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")

# Bounded so prompt size (and latency, already the scarce resource here)
# doesn't grow without limit over a long session -- last 4 exchanges is
# enough context for "yes"/"no"/short-correction replies to resolve against
# without re-litigating an entire conversation on every turn.
_MAX_HISTORY_TURNS = 8


def _append_history(
    history: list[ChatTurn], user_text: str, assistant_text: str | None
) -> list[ChatTurn]:
    new = list(history)
    new.append(ChatTurn(role="user", content=user_text))
    if assistant_text:
        new.append(ChatTurn(role="assistant", content=assistant_text))
    return new[-_MAX_HISTORY_TURNS:]


def _extract_number(text: str) -> float | int | None:
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    n = float(m.group(1))
    return int(n) if n.is_integer() else n


_YES_WORDS = {"yes", "yeah", "yep", "yup", "correct", "right", "confirm",
              "confirmed", "sure", "ok", "okay", "y"}
_NO_WORDS = {"no", "nope", "nah", "n", "incorrect", "wrong"}


def _extract_yes_no(text: str) -> bool | None:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    # Only trust this for a SHORT reply that's basically just the word
    # itself (e.g. "Yes", "yeah sure") -- a longer sentence containing
    # "yes" incidentally isn't necessarily a plain confirmation.
    if len(words) > 3:
        return None
    if any(w in _YES_WORDS for w in words) and not any(w in _NO_WORDS for w in words):
        return True
    if any(w in _NO_WORDS for w in words) and not any(w in _YES_WORDS for w in words):
        return False
    return None


def _extract_unit(text: str) -> str | None:
    low = text.lower()
    if "month" in low:
        return "months"
    if "day" in low:
        return "days"
    if "year" in low:
        return "years"
    return None


def _canonicalize_country_filter(f: Filter) -> Filter:
    """Resolve a colloquial country name ("USA", "UK") to the exact spelling
    candidates are tagged with -- see candidates.canonicalize_country."""
    if f.field != "country":
        return f
    if isinstance(f.value, str):
        return f.model_copy(update={"value": canonicalize_country(f.value)})
    if isinstance(f.value, list):
        return f.model_copy(update={
            "value": [canonicalize_country(v) if isinstance(v, str) else v for v in f.value]
        })
    return f


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
        logger.info(
            "REQUEST_IN job_id=%s session_id=%s reset=%s query=%r "
            "active_filters=%s pending_clarify=%s pending_combine=%s pending_lookup_field=%s",
            job_id, session_id, reset, query,
            [f.model_dump(exclude_none=True) for f in spec.filters],
            bool(current.pending_clarify), bool(current.pending_combine),
            current.pending_lookup_field,
        )

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
                    history=_append_history(current.history, query, answer),
                ))
                return FilterResponse(
                    status="answer", logic=spec.logic,
                    filters=spec.filters, chips=to_chips(spec.filters),
                    message=answer,
                )
            # Didn't match one of the offered names -- treat as abandoning
            # the pending lookup and fall through to a fresh query below.

        # Same idea for a pending COMBINE confirmation ("Do you want
        # candidates matching Mumbai and High tier college?"): a bare
        # "yes"/"no" resolves deterministically, no LLM call, applying the
        # exact pre-computed merged spec rather than re-deriving it (see
        # PendingCombine's docstring for why this confirmation exists at
        # all).
        if current.pending_combine and len(query.split()) <= 6:
            answer = _extract_yes_no(query)
            if answer is True:
                pc = current.pending_combine
                return self._validate_apply_persist(
                    pc.spec.filters, pc.spec.logic, job_id, session_id, query,
                    current.history, extra_message=pc.message,
                )
            if answer is False:
                question = "Okay -- what would you like instead?"
                self.store.set(session_id, job_id, SessionState(
                    spec=spec, last_candidates=current.last_candidates,
                    history=_append_history(current.history, query, question),
                ))
                return FilterResponse(
                    status="clarify", question=question,
                    logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
                )
            # Not a recognizable yes/no -- treat as abandoning the pending
            # combine and fall through to a fresh query below.

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
            pc = current.pending_clarify

            # CONFIRM-style clarify ("Should the experience be at least 7
            # years?"): the value is already known (see
            # LLMOutput.clarify_value) -- a bare "yes"/"no" resolves
            # entirely in code, no LLM call, so it can't hallucinate a value
            # it was never actually given (confirmed failure mode: qwen3:8b
            # asked for "Yes"/"No" without thinking on invented a false
            # "field not supported" claim instead of just applying the
            # already-known value -- this bypasses needing the LLM to
            # re-derive it at all).
            if pc.value is not None:
                answer = _extract_yes_no(query)
                if answer is True:
                    filt = Filter(field=pc.field, operator=pc.operator,
                                   value=pc.value, skill=pc.skill, unit=pc.unit)
                    merged = merge_filters(spec.filters, [filt])
                    return self._validate_apply_persist(
                        merged, spec.logic, job_id, session_id, query, current.history,
                    )
                if answer is False:
                    question = "Okay -- what should it be instead?"
                    self.store.set(session_id, job_id, SessionState(
                        spec=spec, last_candidates=current.last_candidates,
                        history=_append_history(current.history, query, question),
                    ))
                    return FilterResponse(
                        status="clarify", question=question,
                        logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
                    )
                # Not a recognizable yes/no -- might be a new number
                # ("actually make it 8") or a fresh query; fall through.

            value = _extract_number(query)
            if value is not None:
                filt = Filter(field=pc.field, operator=pc.operator, value=value, skill=pc.skill)
                if pc.field == "notice_period":
                    filt.unit = _extract_unit(query) or "days"
                merged = merge_filters(spec.filters, [filt])
                return self._validate_apply_persist(
                    merged, spec.logic, job_id, session_id, query, current.history,
                )
            # No number in the reply -- treat as abandoning the pending
            # clarification and fall through to a fresh query below.

        history_msgs = [t.model_dump() for t in current.history]
        llm_out: LLMOutput = self.llm.translate(
            query, [f.model_dump(exclude_none=True) for f in spec.filters], history_msgs
        )

        if llm_out.intent == "CLARIFY":
            pending = None
            if llm_out.clarify_field:
                pending = PendingClarify(
                    field=llm_out.clarify_field,
                    operator=llm_out.clarify_operator or "gte",
                    skill=llm_out.clarify_skill,
                    value=llm_out.clarify_value,
                    unit=llm_out.clarify_unit,
                )
            question = llm_out.question or "Could you clarify your filter?"
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                pending_clarify=pending,
                history=_append_history(current.history, query, question),
            ))
            return FilterResponse(
                status="clarify",
                question=question,
                options=llm_out.options,
                logic=spec.logic,
                filters=spec.filters,
                chips=to_chips(spec.filters),
            )

        if llm_out.intent == "UNSUPPORTED_FILTER":
            message = llm_out.message or "That filter is not supported."
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, message),
            ))
            return FilterResponse(
                status="unsupported",
                message=message,
                logic=spec.logic,
                filters=spec.filters,
                chips=to_chips(spec.filters),
            )

        if llm_out.intent == "LOOKUP":
            return self._answer_lookup(
                llm_out, current, spec, job_id, session_id, query,
            )

        if llm_out.intent == "EXPERIENCE_SEARCH":
            return self._answer_experience_search(
                llm_out, current, spec, job_id, session_id, query,
            )

        # FILTER_CANDIDATES. Expand skill concepts ("machine learning" ->
        # its real tools) against the curated taxonomy before merging into
        # session state, so what gets stored/matched is already the precise
        # expansion, not the bare concept term. Also resolve any colloquial
        # country name ("USA", "UK") to the exact spelling candidates are
        # tagged with -- a fixed lookup table, not something worth asking
        # the LLM to memorize (see candidates.canonicalize_country).
        expanded_filters = expand_skill_filters([
            _canonicalize_country_filter(f) for f in llm_out.filters
        ])
        effective = (
            expanded_filters if llm_out.replace_all
            else merge_filters(spec.filters, expanded_filters)
        )

        # A query that COMBINES a new field with the existing search is
        # ambiguous enough to confirm rather than silently apply -- see
        # PendingCombine's docstring. Checked against the EFFECTIVE result,
        # not the replace_all flag directly -- confirmed live that flag is
        # unreliable on its own (the model sometimes sets replace_all=True
        # but *also* re-lists the old filters itself, landing on the same
        # combined result merge_filters would have produced anyway). What
        # actually matters is whether every old field survives into the
        # new set (a genuine addition) as opposed to being dropped (a
        # clean replace, e.g. "candidates in mumbai" replacing a stale
        # unrelated search -- see rule 1b) -- that's the real ambiguity,
        # regardless of which code path produced it.
        existing_fields = {f.field for f in spec.filters}
        effective_fields = {f.field for f in effective}
        is_combining = (
            bool(spec.filters)
            and existing_fields.issubset(effective_fields)
            and bool(effective_fields - existing_fields)
        )
        if is_combining:
            proposed_spec = FilterSpec(logic=llm_out.logic, filters=effective)
            labels = [
                re.sub(r"^\S+\s*", "", c.label) for c in to_chips(effective)
            ]
            question = f"Do you want candidates matching {' and '.join(labels)}?"
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                pending_combine=PendingCombine(spec=proposed_spec, message=llm_out.message),
                history=_append_history(current.history, query, question),
            ))
            return FilterResponse(
                status="clarify", question=question, options=["Yes", "No"],
                logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
            )

        return self._validate_apply_persist(
            effective, llm_out.logic, job_id, session_id, query, current.history,
            extra_message=llm_out.message,
        )

    # ------------------------------------------------------------------ #
    # LOOKUP: a question about one already-shown candidate, not a new
    # filter. Resolved entirely deterministically against real stored data
    # -- the LLM only identified WHICH question, never the answer itself.
    # ------------------------------------------------------------------ #
    def _answer_lookup(
        self, llm_out: LLMOutput, current: SessionState, spec: FilterSpec,
        job_id: str, session_id: str, query: str,
    ) -> FilterResponse:
        base = dict(
            status="unsupported", logic=spec.logic,
            filters=spec.filters, chips=to_chips(spec.filters),
        )

        if not llm_out.lookup_field:
            message = ("I wasn't sure what you were asking about that "
                       "candidate -- could you rephrase?")
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, message),
            ))
            return FilterResponse(**base, message=message)

        resolution = resolve_candidate(llm_out.candidate_ref, current.last_candidates)

        if resolution.candidate is None and not resolution.ambiguous_names:
            message = ("I don't have a candidate in view to answer that about "
                       "-- search for someone first.")
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, message),
            ))
            return FilterResponse(**base, message=message)

        if resolution.candidate is None:
            # Remember what was being asked, so the next message (a bare
            # name, typed or clicked) can complete it directly.
            question = f"Which candidate did you mean -- {', '.join(resolution.ambiguous_names)}?"
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                pending_lookup_field=llm_out.lookup_field,
                history=_append_history(current.history, query, question),
            ))
            return FilterResponse(
                status="clarify",
                logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
                question=question,
                options=resolution.ambiguous_names,
            )

        answer = answer_lookup(resolution.candidate, llm_out.lookup_field)
        self.store.set(session_id, job_id, SessionState(
            spec=spec, last_candidates=current.last_candidates,
            history=_append_history(current.history, query, answer),
        ))
        return FilterResponse(
            status="answer",
            logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
            message=answer,
        )

    # Similarity floor for EXPERIENCE_SEARCH -- deliberately a SEPARATE,
    # higher constant from semantic.MIN_SIMILARITY (0.55), not a reuse of
    # it. That value was calibrated against short skill-list text; full
    # job-description sentences are a different kind of text with a
    # different, noisier score distribution on this model. Confirmed
    # empirically across two independent real queries ("led a team of
    # engineers", "built a payment processing system"): genuine matches
    # cluster in the top ~8-10 results (0.62-0.67), but by rank ~15-20
    # (still 0.61-0.62) clearly unrelated text is already interleaved in
    # -- e.g. a plain "Systems Engineer... provided hardware support" (no
    # leadership at all) scored HIGHER (0.6157) than a genuine "Team
    # Leader" title-only entry (0.5958) for the team-leading query. There
    # is no clean cliff in this data -- 0.60 is where both queries' "mostly
    # genuine" and "mostly noise" zones roughly divide, not a perfect cut.
    _EXPERIENCE_MIN_SIMILARITY = 0.60

    # ------------------------------------------------------------------ #
    # EXPERIENCE_SEARCH: the query describes an action/achievement, not a
    # named field -- matched against the actual sentences of each
    # candidate's real job history via semantic search over
    # experience_index (see that module's docstring), not a structured
    # filter. Never invents a match: a candidate only appears here because
    # their own real description text scored close to the query.
    # ------------------------------------------------------------------ #
    def _answer_experience_search(
        self, llm_out: LLMOutput, current: SessionState, spec: FilterSpec,
        job_id: str, session_id: str, query: str,
    ) -> FilterResponse:
        base = dict(logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters))

        if not llm_out.experience_query:
            message = ("I wasn't sure what to search for -- could you describe "
                       "what they should have done?")
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, message),
            ))
            return FilterResponse(status="unsupported", message=message, **base)

        if not experience_index.IndexPaths().exists():
            message = ("Experience-based search isn't available yet -- try a "
                       "skill, title, or company filter instead.")
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, message),
            ))
            return FilterResponse(status="unsupported", message=message, **base)

        # Scope to this job's real matched candidates, AND (if a structured
        # search is already active this session) to those who already pass
        # it -- so "Python devs who led a team" works as two turns: skill
        # narrows first, this intersects semantically on top of that,
        # rather than searching the whole job pool from scratch.
        candidates = get_matched_candidates(job_id)
        pool = apply_spec(candidates, spec) if spec.filters else candidates
        pool_ids = {c.get("id") for c in pool}
        by_id = {c.get("id"): c for c in pool}

        try:
            hits = experience_index.search(llm_out.experience_query, top_k=200)
        except Exception:
            logger.warning(
                "experience_index.search failed for %r", llm_out.experience_query,
                exc_info=True,
            )
            hits = []

        # A candidate can have multiple matching experiences (chunks) --
        # keep their single best score, restricted to the real, in-scope
        # pool computed above (never a candidate outside this job/filter
        # set), and below _EXPERIENCE_MIN_SIMILARITY (see that constant's
        # docstring for the empirical basis) a "match" is noise, not a
        # genuine hit.
        best_score: dict[str, float] = {}
        for hit in hits:
            cid = hit.get("candidate_id")
            if cid not in pool_ids:
                continue
            score = float(hit.get("score", 0.0))
            if score < self._EXPERIENCE_MIN_SIMILARITY:
                continue
            if score > best_score.get(cid, -1.0):
                best_score[cid] = score

        matched = []
        for cid, score in sorted(best_score.items(), key=lambda kv: -kv[1]):
            enriched = dict(by_id[cid])
            enriched["experience_match_score"] = round(score, 4)
            matched.append(enriched)

        logger.info(
            "EXPERIENCE_SEARCH job_id=%s query=%r pool=%d raw_hits=%d "
            "floor=%.2f kept=%d names=%s",
            job_id, llm_out.experience_query, len(pool), len(hits),
            self._EXPERIENCE_MIN_SIMILARITY, len(matched),
            [(c["name"], c["experience_match_score"]) for c in matched[:20]],
        )

        chips = [Chip(label=f'\U0001f50e "{llm_out.experience_query}"', field="experience_query")]
        summary = (
            f'Found {len(matched)} matching "{llm_out.experience_query}"' if matched
            else f'No one matched "{llm_out.experience_query}"'
        )
        self.store.set(session_id, job_id, SessionState(
            spec=spec, last_candidates=matched,
            history=_append_history(current.history, query, summary),
        ))

        if not matched:
            return FilterResponse(
                status="no_match",
                total=len(candidates), showing=0,
                logic=spec.logic, filters=spec.filters, chips=chips,
                message=f'No candidates\' work history matched "{llm_out.experience_query}".',
            )

        return FilterResponse(
            status="ok",
            total=len(candidates), showing=len(matched),
            logic=spec.logic, filters=spec.filters, chips=chips,
            candidates=matched,
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
        query: str | None = None, history: list[ChatTurn] | None = None,
        extra_message: str | None = None,
    ) -> FilterResponse:
        available = get_available_fields(job_id)
        result = validate_filters(filters, available)
        history = history if history is not None else []
        logger.info(
            "VALIDATE job_id=%s input_filters=%s -> ok=%s validated=%s skipped=%s "
            "unsupported=%s error=%r",
            job_id, [f.model_dump(exclude_none=True) for f in filters],
            result.ok, [f.model_dump(exclude_none=True) for f in result.filters],
            result.skipped, result.unsupported, result.error,
        )

        if not result.ok:
            status = "unsupported" if result.unsupported else "error"
            # Persist only the filters that were valid before the bad one?
            # Safer: do not mutate stored state on invalid input -- but the
            # conversation still happened, so still remember it (a later
            # short reply may refer back to this rejection).
            current = self.store.get(session_id, job_id)
            if query is not None:
                self.store.set(session_id, job_id, SessionState(
                    spec=current.spec, last_candidates=current.last_candidates,
                    pending_clarify=current.pending_clarify,
                    pending_lookup_field=current.pending_lookup_field,
                    history=_append_history(history, query, result.error),
                ))
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
        logger.info(
            "APPLY_SPEC job_id=%s pool=%d logic=%s exact_matches=%d names=%s",
            job_id, len(candidates), spec.logic, len(filtered),
            [c.get("name") for c in filtered[:20]],
        )

        # Widen skill filters to also count a candidate who has the same
        # thing in different words -- a curated related tool (merged_tools.json)
        # or, failing that, an LLM-verified semantically-equivalent skill list
        # (see _fuzzy_skill_matches). Full matches merge directly, same
        # ranking, no special labeling. A candidate who satisfies only SOME
        # of several AND'd skill requirements is never silently dropped --
        # they're appended below the full matches, tagged with exactly what
        # they matched and what they didn't (partial_skill_match), so the
        # recruiter sees and judges them instead of the system hiding a
        # possibly-relevant person.
        full_extra, partial = self._fuzzy_skill_matches(
            job_id, spec, matched_ids={c.get("id") for c in filtered},
        )
        if full_extra:
            filtered = sorted(
                filtered + full_extra, key=lambda c: c.get("match_score", 0), reverse=True,
            )
        if partial:
            partial.sort(key=lambda c: (
                -c["partial_skill_match"]["matched"], -c.get("match_score", 0),
            ))
            filtered = filtered + partial
        if full_extra or partial:
            logger.info(
                "FUZZY_SKILL_MATCH job_id=%s full_extra=%d(%s) partial=%d(%s)",
                job_id, len(full_extra), [c.get("name") for c in full_extra],
                len(partial), [c.get("name") for c in partial],
            )

        # Persist the new valid state, including who's now in view -- this
        # is what a later LOOKUP ("which college did he go to") resolves
        # against.
        assistant_summary = (
            f"Applied filters: {', '.join(c.label for c in to_chips(spec.filters))}"
            if spec.filters else "Cleared all filters"
        )
        self.store.set(session_id, job_id, SessionState(
            spec=spec, last_candidates=filtered,
            history=(
                _append_history(history, query, assistant_summary)
                if query is not None else history
            ),
        ))

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
        logger.info(
            "RESULT job_id=%s status=%s total=%d showing=%d skip_note=%r",
            job_id, "no_match" if not filtered else "ok",
            len(candidates), len(filtered), skip_note,
        )

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

    # ------------------------------------------------------------------ #
    # Widen skill filters so a candidate counts as a match via a curated
    # related tool or an LLM-verified semantically-equivalent skill list,
    # not just the exact word -- merged directly into the real result.
    #
    # Under AND logic with 2+ skill filters, a candidate who satisfies SOME
    # but not all of them is still surfaced (never silently dropped) as a
    # PARTIAL match -- ranked below full matches, tagged with exactly which
    # requirements they met and which they didn't, so the recruiter decides
    # rather than the system hiding a possibly-relevant person. Every other
    # filter (non-skill fields, and the skill filters under OR/NOT logic)
    # stays a strict, unlabeled hard requirement -- only skill-under-AND
    # gets partial credit, since that's the specific "same meaning,
    # different words" gap this feature addresses.
    # ------------------------------------------------------------------ #
    _SEMANTIC_SHORTLIST_SIZE = 8

    def _fuzzy_skill_matches(
        self, job_id: str, spec: FilterSpec, matched_ids: set,
    ) -> tuple[list[dict], list[dict]]:
        """Returns (full_extra, partial) -- full_extra: candidates who now
        satisfy the ENTIRE spec (merge directly, no distinction from an
        exact match). partial: candidates who satisfy every non-skill
        filter plus SOME (not all) skill filters under AND logic -- each
        dict carries a `partial_skill_match` key: {"matched": int, "total":
        int, "missing": [label, ...]} for card-level display."""
        skill_idx = [
            i for i, f in enumerate(spec.filters)
            if f.field == "skill" and f.operator in {"contains", "not_contains"}
        ]
        if not skill_idx:
            return [], []

        candidates = get_matched_candidates(job_id)
        pool = [c for c in candidates if c.get("id") not in matched_ids]
        if not pool:
            return [], []

        # Per skill filter: the set of candidate ids (from `pool`) that
        # qualify via EITHER a curated taxonomy relation OR an LLM-verified
        # semantically-equivalent skill list -- raw embedding similarity
        # alone is deliberately never trusted on its own here (see
        # skill_verify.py's docstring: confirmed a cybersecurity candidate
        # scored within 0.006 of a genuine ML match on pure similarity).
        qualifying_by_filter: dict[int, set] = {}
        for i in skill_idx:
            canon = canonicalize(str(spec.filters[i].value))
            exact, related = related_terms_for(canon)

            qualifies = set()
            for c in pool:
                cand_skills = {s.lower() for s in skill_names_of(c)}
                if cand_skills & exact or cand_skills & related:
                    qualifies.add(c.get("id"))

            try:
                sims = term_similarities(job_id, canon)
            except Exception:
                logger.warning("term_similarities failed for %r", canon, exc_info=True)
                sims = {}
            remaining = [c for c in pool if c.get("id") not in qualifies]
            shortlist_candidates = sorted(
                (c for c in remaining if sims.get(c.get("id"), 0.0) >= MIN_SIMILARITY),
                key=lambda c: -sims.get(c.get("id"), 0.0),
            )[: self._SEMANTIC_SHORTLIST_SIZE]
            if shortlist_candidates:
                shortlist = [(c.get("id"), skill_names_of(c)) for c in shortlist_candidates]
                try:
                    qualifies |= verify_skill_candidates(canon, shortlist)
                except Exception:
                    logger.warning("verify_skill_candidates failed for %r", canon, exc_info=True)

            qualifying_by_filter[i] = qualifies

        skill_labels = {i: canonicalize(str(spec.filters[i].value)) for i in skill_idx}

        full_extra, partial = [], []
        for c in pool:
            cid = c.get("id")
            non_skill_ok = True
            skill_hits, skill_misses = [], []
            for i, f in enumerate(spec.filters):
                if i in qualifying_by_filter:
                    hit = cid in qualifying_by_filter[i]
                    if f.operator == "not_contains":
                        hit = not hit
                    (skill_hits if hit else skill_misses).append(skill_labels[i])
                else:
                    non_skill_ok = non_skill_ok and matches_filter(c, f)

            if spec.logic == "OR":
                if non_skill_ok or skill_hits:
                    full_extra.append(c)
            elif spec.logic == "NOT":
                if not non_skill_ok and not skill_hits:
                    full_extra.append(c)
            else:  # AND
                if not non_skill_ok:
                    continue
                if not skill_misses:
                    full_extra.append(c)
                elif skill_hits:
                    enriched = dict(c)
                    enriched["partial_skill_match"] = {
                        "matched": len(skill_hits),
                        "total": len(skill_hits) + len(skill_misses),
                        "missing": skill_misses,
                    }
                    partial.append(enriched)
        return full_extra, partial


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
