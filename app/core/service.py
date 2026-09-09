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
from app.core.merge import alternative_group_chips, merge_filters, preferred_chips, to_chips
from app.core.session import SessionStore, default_store
from app.core.skill_taxonomy import (
    canonicalize, expand_skill_filters, related_terms_for, skill_names_of,
)
from app.core.semantic import MIN_SIMILARITY, term_similarities
from app.core.validation import validate_filters
from app.llm.client import LLMClient
from app.llm.skill_verify import verify_skill_candidates
from app.models.schemas import (
    AlternativeGroup,
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


def _all_chips(spec: FilterSpec) -> list[Chip]:
    """Full chip set for a spec -- ordinary required filters, alternative
    groups, and preferred (non-exclusionary) filters together, in that
    order."""
    return (
        to_chips(spec.filters)
        + alternative_group_chips(spec.alternative_groups)
        + preferred_chips(spec.preferred_filters)
    )


def _extract_unit(text: str) -> str | None:
    low = text.lower()
    if "month" in low:
        return "months"
    if "day" in low:
        return "days"
    if "year" in low:
        return "years"
    return None


# Deterministic safety net for a confirmed, repeatable model failure (see
# prompt.py rule 7a): a query naming COMPANY SIZE ("large company", "big
# company", ...) reliably got a fabricated "company_type" filter tacked on
# ALONGSIDE the correct "company size isn't tracked" message -- confirmed
# live, byte-identical, across 6 separate live attempts even after two
# rounds of prompt strengthening (two dedicated few-shot examples, then an
# explicit rule stating the principle directly). Prompt wording alone does
# not reliably stop this for this model, so it's enforced here instead --
# consistent with this codebase's standing approach elsewhere (PendingClarify,
# validation.py, the mandatory Unknown category, ...): never rely on the LLM
# alone for a guarantee that actually matters, when a deterministic check can
# make it certain instead.
_COMPANY_SIZE_WORDS_RE = re.compile(
    r"\b(large|big|huge|massive|sizable|giant|small|mid[- ]?size[d]?)\s+compan"
    r"|\bcompany\s+size\b",
    re.IGNORECASE,
)


def _dedupe_hard_filters_duplicated_elsewhere(
    filters: list[Filter],
    preferred_filters: list[Filter],
    alternative_groups: list[AlternativeGroup],
) -> list[Filter]:
    """Confirmed live TWICE, same failure shape both times: the model
    correctly puts a requirement somewhere it should be non-exclusionary or
    optional (preferred_filters, or one branch of an alternative_group) AND
    ALSO, separately, leaves a hard duplicate of that exact same (field,
    value) in the ordinary `filters` list. The hard duplicate silently
    defeats the entire point either way:
    - a "preferred" criterion that's ALSO hard-required isn't a preference
      anymore -- it excludes exactly the people "prefer" was meant to still
      include.
    - a branch of an "either X or Y" alternative that's ALSO hard-required
      on its own forces BOTH alternatives at once (plus everything else in
      that branch), recreating the exact all-required bug alternative_groups
      exists to prevent -- confirmed live: "either a Master's from a Tier-1
      university or 10+ years" correctly built the right alternative_group
      structure, but the SAME three filters (education, college_tier,
      experience) also stayed in the hard list, so the group's flexibility
      never mattered -- everything was required anyway.
    If the model already correctly placed a requirement somewhere with
    different (non-exclusionary, or alternative) semantics, a hard
    duplicate of that exact same requirement is never correct -- drop it,
    the other copy already covers it correctly."""
    protected_keys = {(f.field, str(f.value).lower()) for f in preferred_filters}
    for group in alternative_groups:
        for branch in group.branches:
            protected_keys.update((f.field, str(f.value).lower()) for f in branch)
    return [f for f in filters if (f.field, str(f.value).lower()) not in protected_keys]


_COORD_OR_RE = re.compile(r"\s+or\s+", re.IGNORECASE)


def _expand_coordinated_job_title(f: Filter) -> Filter:
    """Deterministic safety net for a confirmed, repeated failure: an
    elided-head-noun either/or role phrase ("senior or lead backend
    engineer", meaning "senior backend engineer OR lead backend engineer")
    keeps getting emitted as ONE literal "contains" filter with the whole
    phrase mashed together -- which then requires a resume to contain that
    exact unlikely string verbatim, matching nobody. This is the job_title
    analogue of rule 3c (same-field alternatives -> operator "in"), except
    the alternatives here are elided rather than fully spelled out, which
    needs actual expansion, not just detection -- confirmed live that
    reinforcing rule 3c's existing few-shot examples with an explicit
    job_title case did not reliably fix this shape at full compound-query
    complexity (same lesson as the alternative_groups reinforcement
    elsewhere in this module: prompt-only fixes for a well-defined pattern
    class cap out, so enforce it in code instead).

    Only fires on a "contains"/"not_contains" job_title filter whose value
    contains exactly one standalone " or " -- multiple "or"s make the
    coordination ambiguous enough that guessing would be worse than leaving
    it alone. Heuristic: split on " or "; if the LEFT side has fewer words
    than the right, assume left is a bare modifier missing the head noun,
    and borrow that head from the right side's tail (e.g. "senior" + "lead
    backend engineer" -> "senior backend engineer" / "lead backend
    engineer") -- this leading-modifier shape ("senior or lead X", "junior
    or mid-level X") is the form recruiter phrasing actually uses. A
    trailing-modifier shape (head noun first, e.g. "X, senior or lead") is
    not attempted -- which word span is the head vs. the modifier is
    genuinely ambiguous from word count alone in that direction, and a
    confidently-wrong split would be worse than leaving the filter as one
    (already-broken, but not newly-wrong) mashed phrase. Equal word counts
    (e.g. "software engineer or data scientist") need no expansion -- both
    sides are already complete phrases, used as-is."""
    if f.field != "job_title" or f.operator not in ("contains", "not_contains"):
        return f
    value = str(f.value)
    parts = _COORD_OR_RE.split(value)
    if len(parts) != 2:
        return f
    left, right = (p.strip() for p in parts)
    if not left or not right:
        return f
    left_words, right_words = left.split(), right.split()
    if len(left_words) < len(right_words):
        head = " ".join(right_words[len(left_words):])
        left = f"{left} {head}"
    new_op = "not_in" if f.operator == "not_contains" else "in"
    return Filter(field="job_title", operator=new_op, value=[left, right])


def _expand_coordinated_job_titles(filters: list[Filter]) -> list[Filter]:
    return [_expand_coordinated_job_title(f) for f in filters]


def _strip_fabricated_company_size_filters(
    query: str, filters: list[Filter], message: str | None,
) -> tuple[list[Filter], str | None]:
    """If NEW QUERY talks about company SIZE, company_type/company_tier
    filters cannot have come from anything real in the query (size is a
    genuinely different, untracked axis from either) -- drop them and make
    sure the recruiter still gets an honest note, regardless of whether the
    model's own message already included one."""
    if not _COMPANY_SIZE_WORDS_RE.search(query):
        return filters, message
    kept, dropped = [], False
    for f in filters:
        if f.field in ("company_type", "company_tier"):
            dropped = True
            continue
        kept.append(f)
    if not dropped:
        return filters, message
    note = "Company size isn't tracked, so that part couldn't be applied."
    if message and "company size" not in message.lower():
        message = f"{message} {note}"
    elif not message:
        message = note
    return kept, message


# Deterministic safety net for a DIFFERENT confirmed, repeatable model
# failure -- confirmed live in THREE separate, differently-shaped
# conversations: given a NEW QUERY that shares no real content with the
# already-active filters, the model sometimes just re-emits those same old
# filters verbatim instead of translating the new query at all (e.g.
# CURRENT FILTERS = job_title/domain/skill about "gaming", NEW QUERY =
# "candidates with data science and BI skills" -> returned the SAME
# gaming/C++ filters, unchanged). Two rounds of prompt rules (1c, then the
# sneakier self-reinforcement case in 1d) measurably reduced how often this
# happens but did not eliminate it -- confirmed live a 4th time, in a full
# regression run, even with both rules in place. Same lesson as the
# company-size guard above: once a prompt-only fix has been tried and shown
# to still fail unpredictably, enforce the guarantee in code instead of
# hoping the next wording tweak is the one that finally sticks.
def _looks_like_stale_filter_echo(
    query: str, current_filters: list[Filter], returned_filters: list[Filter],
) -> bool:
    """True only when BOTH hold: (1) the filters the model just returned are
    IDENTICAL (same field+value pairs) to what was already active, and (2)
    NONE of those filters' own values are mentioned anywhere in NEW QUERY --
    meaning nothing about the new question could legitimately have produced
    the same answer again. A real, intentional repeat (the recruiter
    actually re-asking for the same thing) almost always still names at
    least part of what they want, which is what keeps this from false-
    firing on genuine repeats -- only a NEW query with literally no
    connection to the stale filters trips it."""
    if not current_filters or not returned_filters:
        return False
    def key_set(filters: list[Filter]) -> set[tuple]:
        return {(f.field, str(f.value).lower()) for f in filters}
    if key_set(current_filters) != key_set(returned_filters):
        return False
    query_lower = query.lower()
    for f in current_filters:
        values = f.value if isinstance(f.value, list) else [f.value]
        if any(str(v).lower() in query_lower for v in values):
            return False
    return True


def _filter_survived_unchanged(f: Filter, effective: list[Filter]) -> bool:
    """True if `f` (a filter that was already active) is still present in
    `effective` with the EXACT SAME value -- not merely the same field.
    Distinguishes a genuine "keep this AND also add that" combination from
    a "no, I meant something different for this field" replacement (e.g.
    "software engineer" replacing "backend engineer") -- only the former
    is the real ambiguity PendingCombine exists for (see its own
    docstring); the latter is an unambiguous, self-contained request that
    should just apply, the way any real chat assistant handles a complete
    new sentence instead of asking the user to re-approve their own
    words."""
    return any(
        other.field == f.field and str(other.value).lower() == str(f.value).lower()
        for other in effective
    )


_EITHER_OR_RE = re.compile(r"\beither\b.*\bor\b", re.IGNORECASE)


def _fix_missed_or_logic(query: str, logic: str, filters: list[Filter]) -> str:
    """Deterministic self-heal for a confirmed, reproducible LLM logic
    mistake: "candidates who have either X or Y" -- a standalone same-field
    either/or with nothing else in the query -- sometimes comes back with
    "logic":"AND" instead of "OR". Confirmed on a real eval run: a query
    BYTE-IDENTICAL to prompt.py's own worked example for this exact case
    (rule 4's first worked example) still failed -- same lesson as every
    other deterministic fix in this module: prompt reinforcement alone
    isn't reliable here, even with a literal matching example already
    present.

    Deliberately narrow: only fires when the ENTIRE result is exactly two
    filters on the SAME field and operator (the shape rule 4's first case
    describes). A query where "either X or Y" is only one piece of a
    larger AND'd request (rule 4's second case, expressed via
    alternative_groups instead, e.g. "Python developers ... with either
    Kubernetes or Terraform") naturally has more than two filters, or two
    filters on DIFFERENT fields, and is correctly left untouched -- same as
    a query with no "either...or" phrasing at all."""
    if logic != "AND" or len(filters) != 2:
        return logic
    if not _EITHER_OR_RE.search(query):
        return logic
    f1, f2 = filters
    if f1.field == f2.field and f1.operator == f2.operator:
        return "OR"
    return logic


_UNTRACKED_LOOKUP_RE = re.compile(
    r"\b(email|e-mail|phone(?: number)?|contact(?: number| info| details| information)?|"
    r"resume|cv|linkedin|github)\b",
    re.IGNORECASE,
)


def _is_untracked_lookup(query: str) -> bool:
    """Deterministic safety net for a confirmed, reproducible LLM intent
    mistake: a question about contact details ("what's his email address")
    sometimes comes back as intent LOOKUP instead of UNSUPPORTED_FILTER.
    Confirmed on a real eval run: a query BYTE-IDENTICAL to prompt.py's own
    worked example for this exact case (rule 6a-ii's final sentence) still
    failed. LOOKUP is resolved entirely against real stored candidate data
    (see lookup.answer_lookup) -- there is no real field any of these
    concepts could resolve into, so letting LOOKUP through risks the
    backend trying to answer from whatever field the model guessed instead
    of honestly saying the data isn't tracked at all."""
    return bool(_UNTRACKED_LOOKUP_RE.search(query))


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

        if llm_out.intent == "LOOKUP" and _is_untracked_lookup(query):
            message = (
                "Contact details aren't tracked -- only location, "
                "experience, education, university, company, and skills."
            )
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, message),
            ))
            return FilterResponse(
                status="unsupported", message=message,
                logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
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
        expanded_filters = _expand_coordinated_job_titles(expanded_filters)
        expanded_filters, llm_message = _strip_fabricated_company_size_filters(
            query, expanded_filters, llm_out.message,
        )
        llm_out.logic = _fix_missed_or_logic(query, llm_out.logic, expanded_filters)

        # See _looks_like_stale_filter_echo's docstring -- a confirmed,
        # repeated failure mode (3 separate live reproductions) where the
        # model just hands back the CURRENTLY active filters unchanged for
        # a NEW query that has nothing to do with them, instead of either
        # translating it or asking. Never apply that silently -- an honest
        # "I'm not sure I understood that" is always better than a
        # confident-looking result that quietly ignored what was actually
        # asked (same principle as every UNSUPPORTED_FILTER/CLARIFY case
        # elsewhere in this module).
        if _looks_like_stale_filter_echo(query, spec.filters, expanded_filters):
            question = (
                "I'm not sure I understood that -- could you rephrase what "
                "you're looking for?"
            )
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, question),
            ))
            return FilterResponse(
                status="clarify", question=question,
                logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
            )

        effective = (
            expanded_filters if llm_out.replace_all
            else merge_filters(spec.filters, expanded_filters)
        )

        # A query that COMBINES a genuinely NEW field with an otherwise-
        # UNTOUCHED existing search is ambiguous enough to confirm rather
        # than silently apply -- see PendingCombine's docstring. Checked
        # against the EFFECTIVE result, not the replace_all flag directly
        # -- confirmed live that flag is unreliable on its own (the model
        # sometimes sets replace_all=True but *also* re-lists the old
        # filters itself, landing on the same combined result
        # merge_filters would have produced anyway).
        #
        # "Untouched" is decided by exact VALUE survival (see
        # _filter_survived_unchanged), not just field-NAME survival --
        # checking field names alone asks the recruiter to re-approve a
        # complete, self-contained sentence they already typed, which no
        # real chat assistant does. Confirmed live: "show me software
        # engineers ... using java and spring boot" (a fresh, complete
        # request, replace_all=True, correctly dropping the old "Backend
        # Engineer" job_title entirely) still triggered "Do you want
        # candidates matching Software Engineer and ecommerce and Java and
        # Spring Boot?" -- listing nothing but what the recruiter had just
        # explicitly said -- purely because "job_title" as a FIELD NAME
        # still appeared in the new result too, just holding a completely
        # different value ("Software Engineer" replacing "Backend
        # Engineer"). A field-name match alone doesn't mean anything old
        # survived; only an unchanged VALUE match does -- that's the
        # actual ambiguity this confirmation exists for.
        survived_unchanged = [f for f in spec.filters if _filter_survived_unchanged(f, effective)]
        effective_fields = {f.field for f in effective}
        is_combining = (
            bool(spec.filters)
            and len(survived_unchanged) == len(spec.filters)
            and bool(effective_fields - {f.field for f in spec.filters})
        )
        if is_combining:
            proposed_spec = FilterSpec(logic=llm_out.logic, filters=effective)
            labels = [
                re.sub(r"^\S+\s*", "", c.label) for c in to_chips(effective)
            ]
            question = f"Do you want candidates matching {' and '.join(labels)}?"
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                pending_combine=PendingCombine(spec=proposed_spec, message=llm_message),
                history=_append_history(current.history, query, question),
            ))
            return FilterResponse(
                status="clarify", question=question, options=["Yes", "No"],
                logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
            )

        # alternative_groups/preferred_filters merge as coarse, atomic units
        # (not field-by-field like ordinary filters) -- a fresh standalone
        # search (replace_all) uses exactly what the model just returned,
        # even if empty; a refinement turn that names NEW ones replaces the
        # old set with those, but if it names none, the existing ones
        # survive unchanged (same "don't silently drop what wasn't
        # mentioned" principle as everything else in this module).
        effective_alt_groups = (
            llm_out.alternative_groups if (llm_out.replace_all or llm_out.alternative_groups)
            else spec.alternative_groups
        )
        effective_preferred = (
            llm_out.preferred_filters if (llm_out.replace_all or llm_out.preferred_filters)
            else spec.preferred_filters
        )
        # See _dedupe_hard_filters_duplicated_elsewhere's docstring -- never
        # let a hard-required copy of a "preferred" criterion, or of a
        # branch filter inside an alternative_group, silently cancel out
        # the whole point of it being a preference / an alternative.
        effective = _dedupe_hard_filters_duplicated_elsewhere(
            effective, effective_preferred, effective_alt_groups,
        )

        return self._validate_apply_persist(
            effective, llm_out.logic, job_id, session_id, query, current.history,
            extra_message=llm_message,
            alternative_groups=effective_alt_groups,
            preferred_filters=effective_preferred,
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
        # A compound ask ("backend engineers who worked on X") can name a
        # real structured requirement ALONGSIDE the achievement in the same
        # turn -- process any `filters` the model included exactly like a
        # FILTER_CANDIDATES turn would (taxonomy expansion, coordinated
        # job-title splitting, country canonicalization, validation), then
        # merge them into the active session spec so they narrow the pool
        # alongside the semantic search, show up as real chips, and persist
        # for later turns -- rather than being silently dropped just
        # because this turn's intent is EXPERIENCE_SEARCH. Confirmed live:
        # "backend engineer who worked on a supply chain platform" used to
        # search the achievement ALONE, matching non-engineers too (see
        # schemas.py's experience_query docstring for the full history).
        effective_spec = spec
        validation_note = None
        if llm_out.filters:
            expanded = expand_skill_filters([
                _canonicalize_country_filter(f) for f in llm_out.filters
            ])
            expanded = _expand_coordinated_job_titles(expanded)
            result = validate_filters(expanded, get_available_fields(job_id))
            if result.filters:
                effective_spec = FilterSpec(
                    logic=spec.logic, filters=merge_filters(spec.filters, result.filters),
                    alternative_groups=spec.alternative_groups,
                    preferred_filters=spec.preferred_filters,
                )
            if result.skipped:
                validation_note = "Couldn't apply: " + "; ".join(result.skipped) + "."

        base = dict(logic=effective_spec.logic, filters=effective_spec.filters,
                    chips=to_chips(effective_spec.filters))

        if not llm_out.experience_query:
            message = ("I wasn't sure what to search for -- could you describe "
                       "what they should have done?")
            self.store.set(session_id, job_id, SessionState(
                spec=effective_spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, message),
            ))
            return FilterResponse(status="unsupported", message=message, **base)

        if not experience_index.IndexPaths().exists():
            message = ("Experience-based search isn't available yet -- try a "
                       "skill, title, or company filter instead.")
            self.store.set(session_id, job_id, SessionState(
                spec=effective_spec, last_candidates=current.last_candidates,
                history=_append_history(current.history, query, message),
            ))
            return FilterResponse(status="unsupported", message=message, **base)

        # Scope to this job's real matched candidates, AND to those who
        # already pass the active structured spec -- whether it came from
        # an earlier turn or was just merged in above from THIS turn --
        # so a real job_title/skill/location requirement narrows first,
        # and the semantic search intersects on top of that.
        candidates = get_matched_candidates(job_id)
        pool = apply_spec(candidates, effective_spec) if effective_spec.filters else candidates
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
            "EXPERIENCE_SEARCH job_id=%s query=%r structured_filters=%s "
            "pool=%d raw_hits=%d floor=%.2f kept=%d names=%s",
            job_id, llm_out.experience_query,
            [f.model_dump(exclude_none=True) for f in effective_spec.filters],
            len(pool), len(hits), self._EXPERIENCE_MIN_SIMILARITY, len(matched),
            [(c.get("name", "Unnamed"), c["experience_match_score"]) for c in matched[:20]],
        )

        chips = to_chips(effective_spec.filters) + [
            Chip(label=f'\U0001f50e "{llm_out.experience_query}"', field="experience_query")
        ]
        summary = (
            f'Found {len(matched)} matching "{llm_out.experience_query}"' if matched
            else f'No one matched "{llm_out.experience_query}"'
        )
        self.store.set(session_id, job_id, SessionState(
            spec=effective_spec, last_candidates=matched,
            history=_append_history(current.history, query, summary),
        ))

        if not matched:
            no_match_msg = f'No candidates\' work history matched "{llm_out.experience_query}".'
            return FilterResponse(
                status="no_match",
                total=len(candidates), showing=0,
                logic=effective_spec.logic, filters=effective_spec.filters, chips=chips,
                message=" ".join(m for m in (validation_note, no_match_msg) if m),
            )

        return FilterResponse(
            status="ok",
            total=len(candidates), showing=len(matched),
            logic=effective_spec.logic, filters=effective_spec.filters, chips=chips,
            candidates=matched,
            message=validation_note,
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
        alternative_groups: list[AlternativeGroup] | None = None,
        preferred_filters: list[Filter] | None = None,
    ) -> FilterResponse:
        available = get_available_fields(job_id)
        result = validate_filters(filters, available)
        history = history if history is not None else []

        # Validate each branch of each alternative group the SAME way plain
        # filters are validated -- a bad filter inside a branch is dropped
        # from that branch, never silently kept as if it were fine. A
        # branch that loses EVERY filter this way must be dropped entirely
        # (not kept as an empty, vacuously-true branch -- that would let
        # ANY candidate through it, defeating the whole point of it being
        # one alternative among several). A group that loses every branch
        # this way is dropped entirely too -- an alternative group with no
        # real alternatives left isn't a requirement, it's nothing.
        validated_groups: list[AlternativeGroup] = []
        for group in (alternative_groups or []):
            kept_branches = []
            for branch in group.branches:
                branch_result = validate_filters(branch, available)
                if branch_result.filters:
                    kept_branches.append(branch_result.filters)
            if kept_branches:
                validated_groups.append(AlternativeGroup(branches=kept_branches))

        # preferred_filters follow the same field/operator/value validation
        # as ordinary filters -- but NEVER exclude anyone (see below), so an
        # unsupported one is simply dropped, not surfaced as an error.
        preferred_result = validate_filters(preferred_filters or [], available)
        validated_preferred = preferred_result.filters
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

        spec = FilterSpec(
            logic=logic, filters=result.filters,
            alternative_groups=validated_groups, preferred_filters=validated_preferred,
        )
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

        # preferred_filters (see AlternativeGroup/LLMOutput.preferred_filters
        # docstrings) never excludes anyone -- but "prefer" is a real signal,
        # not a no-op, so it becomes a RANKING boost instead: every result is
        # annotated with how many of the preferred criteria it actually
        # meets (for the UI to show, same idea as partial_skill_match), and
        # the "full match" tier is re-sorted so candidates who ALSO meet
        # more preferences rank above otherwise-equal ones. Never touches
        # WHICH candidates appear, only the order among the ones that
        # already qualified on the real requirements. The lower-priority
        # partial-skill-match tier keeps its own existing sort (skill
        # completeness first) -- preference is a tie-breaker refinement,
        # not something that should outrank a genuinely stronger skill match.
        if spec.preferred_filters:
            full_tier_ids = {c.get("id") for c in filtered} - {c.get("id") for c in partial}
            for c in filtered:
                hits = [f for f in spec.preferred_filters if matches_filter(c, f)]
                c["preferred_match"] = {
                    "matched": len(hits), "total": len(spec.preferred_filters),
                }
            filtered = sorted(
                filtered,
                key=lambda c: (
                    0 if c.get("id") in full_tier_ids else 1,
                    -(c["preferred_match"]["matched"] if c.get("id") in full_tier_ids else 0),
                    -c.get("match_score", 0),
                ),
            )

        # Persist the new valid state, including who's now in view -- this
        # is what a later LOOKUP ("which college did he go to") resolves
        # against.
        all_chips = _all_chips(spec)
        assistant_summary = (
            f"Applied filters: {', '.join(c.label for c in all_chips)}"
            if all_chips else "Cleared all filters"
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
                alternative_groups=spec.alternative_groups,
                preferred_filters=spec.preferred_filters,
                chips=all_chips,
                message=skip_note or "No candidates match these filters.",
                suggestions=_no_match_suggestions(spec.filters),
            )

        return FilterResponse(
            status="ok",
            total=len(candidates),
            showing=len(filtered),
            logic=spec.logic,
            alternative_groups=spec.alternative_groups,
            preferred_filters=spec.preferred_filters,
            filters=spec.filters,
            chips=all_chips,
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

    # Below this length, the embedding-similarity step is skipped entirely
    # for a skill term -- confirmed live this is a real, structural gap, not
    # a guess: a bare "AI" query scored a QA/performance-testing candidate
    # with ZERO real AI/ML skills at 0.687 similarity, the single HIGHEST
    # score of any candidate in the job. nomic-embed-text (like most text
    # embedding models) is tuned for words/phrases with real semantic
    # content -- a bare 2-3 character abbreviation ("AI", "ML", "BI", "QA")
    # is too short and ambiguous to embed meaningfully, so the resulting
    # "shortlist" is closer to noise than a real ranking, and even a strong
    # verifier model is then working from garbage candidates. Below this
    # threshold, only exact + curated taxonomy-related matches apply (both
    # unaffected -- this guard is embedding-specific).
    _MIN_SEMANTIC_TERM_LEN = 4

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
        # qualify via an EXACT taxonomy match (the tool's own name/aliases
        # -- always safe, identity-preserving) OR pass LLM verification.
        #
        # A curated "related" taxonomy hit (e.g. Angular for a "React"
        # search) is real signal but is NOT, on its own, an automatic
        # qualifier -- confirmed live this let a candidate with ONLY
        # Angular (no React, no JS-adjacent skill at all) wrongly full-
        # match a "React" search. related_terms_for's own docstring already
        # says why: "related" means genuinely close enough to WIDEN a
        # search, not identical -- it's still a DIFFERENT, sometimes
        # competing tool (skill_taxonomy.py's expand_skill_filters
        # docstring makes the same point for the LLM-output-expansion path:
        # "Python's related tools include Django and SQL; React's include
        # Angular and Vue.js... treating those as OR-equivalent would
        # silently match a candidate who knows a different tool"). This
        # path was the one place in the codebase that didn't yet apply that
        # principle. So: taxonomy-related hits go through the SAME strict
        # LLM verification as an embedding-similarity hit (see
        # skill_verify.py's docstring: raw similarity alone is never
        # trusted either, for the same reason) -- placed first in the
        # shortlist since curated data is more reliable than a bare
        # embedding score, topped up with embedding hits to the size cap.
        qualifying_by_filter: dict[int, set] = {}
        for i in skill_idx:
            canon = canonicalize(str(spec.filters[i].value))
            exact, related = related_terms_for(canon)
            cand_skills = {c.get("id"): {s.lower() for s in skill_names_of(c)} for c in pool}

            qualifies = {cid for cid, skills in cand_skills.items() if skills & exact}
            taxonomy_related = [
                c for c in pool
                if c.get("id") not in qualifies and cand_skills[c.get("id")] & related
            ]

            sims: dict = {}
            if len(canon) >= self._MIN_SEMANTIC_TERM_LEN:
                try:
                    sims = term_similarities(job_id, canon)
                except Exception:
                    logger.warning("term_similarities failed for %r", canon, exc_info=True)

            taxonomy_ids = {c.get("id") for c in taxonomy_related}
            embedding_related = sorted(
                (c for c in pool
                 if c.get("id") not in qualifies and c.get("id") not in taxonomy_ids
                 and sims.get(c.get("id"), 0.0) >= MIN_SIMILARITY),
                key=lambda c: -sims.get(c.get("id"), 0.0),
            )
            shortlist_candidates = (taxonomy_related + embedding_related)[: self._SEMANTIC_SHORTLIST_SIZE]
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
            excluded = False
            skill_hits, skill_misses = [], []
            for i, f in enumerate(spec.filters):
                if i in qualifying_by_filter:
                    qualifies = cid in qualifying_by_filter[i]
                    if f.operator == "not_contains":
                        # A "not_contains" MISS here means the candidate DOES
                        # have the excluded (or a fuzzy-related) skill -- an
                        # absolute disqualifier, not a soft miss to offset
                        # against other skill_hits. Confirmed live: a
                        # candidate excluded via "no PHP" who also happened
                        # to match AWS/Docker was still surfaced in the
                        # PARTIAL tier, because a violated exclusion and a
                        # merely-missed positive requirement were being
                        # counted identically -- "no PHP" means no PHP
                        # developer should ever appear, full stop, same
                        # principle as every other hard-exclusion guard in
                        # this module.
                        if qualifies:
                            excluded = True
                        else:
                            skill_hits.append(skill_labels[i])
                    else:
                        (skill_hits if qualifies else skill_misses).append(skill_labels[i])
                else:
                    non_skill_ok = non_skill_ok and matches_filter(c, f)

            if spec.logic == "OR":
                if non_skill_ok or skill_hits:
                    full_extra.append(c)
            elif spec.logic == "NOT":
                if not non_skill_ok and not skill_hits and not excluded:
                    full_extra.append(c)
            else:  # AND
                if not non_skill_ok or excluded:
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
