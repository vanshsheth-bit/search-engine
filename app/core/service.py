"""Service layer: orchestrates the full pipeline.

query -> LLM translate -> merge with session -> validate -> engine -> response
"""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache

from app.core.candidates import (
    canonicalize_country, experience_texts_by_candidate, get_available_fields,
    get_matched_candidates, mention_pattern,
)
from app.core.engine import apply_spec, matches_filter
from app.core import experience_index
from app.core.lookup import answer_lookup, resolve_candidate
from app.core.merge import chip_label, merge_alternative_groups, merge_filters, to_chips
from app.core.query_log import log_query
from app.core.session import SessionStore, default_store
from app.core.skill_taxonomy import (
    canonicalize, expand_skill_filters, is_known_tool, related_terms_for,
    skill_names_of, tools_for_subdomain,
)
from app.core import taxonomy
from app.core.validation import validate_alternative_groups, validate_filters
from app.core.vocabulary import SENIORITY_BANDS, seniority_band
from app.llm.client import LLMClient
from app.models.schemas import (
    AlternativeGroup,
    ChatTurn,
    Chip,
    Filter,
    FilterChoice,
    FilterResponse,
    FilterSpec,
    LLMOutput,
    PatchStateRequest,
    PendingClarify,
    PendingConfirm,
    PendingDomainSkillPick,
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


def _hard_only(spec: FilterSpec) -> FilterSpec:
    """The view of `spec` that actually EXCLUDES candidates -- drops
    hard=False filters (schema_v2's "nice to have", see Filter.hard) so
    they can't silently become exclusionary again downstream. `spec`
    itself (both hard and soft) is still what gets persisted to
    SessionState and rendered as chips; only the three call sites that
    decide who's IN or OUT of a result (apply_spec in
    _validate_apply_persist, apply_spec in _answer_experience_search's
    pool-scoping, and matches_filter in _fuzzy_skill_matches's non-skill
    check) should ever see this narrowed view.

    Restricted to AND logic on purpose: "hard vs soft, then re-rank the
    survivors" is only a coherent idea when every filter is normally
    required. Under OR, a "soft" filter is a full alternative branch of
    inclusion, not an optional extra -- pulling it out to re-rank
    afterward would change what the query MEANS, not just how results are
    ordered. Under NOT, there's no "survivor" concept to rank at all. So
    for OR/NOT, every filter is treated as hard regardless of its flag --
    simplest correct behavior until compound OR/NOT-with-preferences is a
    real, deliberately-designed feature rather than an accident of this
    helper being applied too eagerly.

    `alternative_groups` (see schemas.AlternativeGroup) has the exact same
    AND-only restriction, for the same reason: "OR of top-level filters"
    composed with "AND-gate on top of that" has no coherent meaning when
    the top-level logic is itself OR/NOT, so it's dropped alongside soft
    filters under those logics.
    """
    if spec.logic != "AND":
        if not spec.alternative_groups:
            return spec
        return spec.model_copy(update={"alternative_groups": []})
    if not any(not f.hard for f in spec.filters):
        return spec
    return spec.model_copy(update={"filters": [f for f in spec.filters if f.hard]})


def _skill_years_available(candidates: list[dict], skill: str | None = None) -> bool:
    """True if at least one candidate in this pool has a REAL (non-None)
    per-skill years number -- for the specific named `skill` (case-
    insensitive) when given, or for any skill at all when `skill` is None.

    Must be checked PER SKILL, not pool-wide: candidates._skill_years_from_
    experience computes a real number for known tools when the resume's own
    job-description text supports it, but NOT for every skill/candidate
    combination -- an unrecognized skill name, or a real tool nobody in this
    specific pool happened to mention in prose, still has nothing. Treating
    "some candidate somewhere has SOME skill's real years" as license to
    trust a DIFFERENT skill's number would silently reintroduce the exact
    bug this function exists to prevent: a skill_experience filter for a
    skill with no real data anywhere in this pool would wrongly stay
    "answerable" and return a false, silent no_match instead of degrading to
    the honest fallback (see the call site in _filter_by_query)."""
    target = skill.lower() if skill else None
    for c in candidates:
        skills = c.get("skills")
        if not isinstance(skills, dict):
            continue
        items = (
            skills.items() if target is None
            else ((k, v) for k, v in skills.items() if k.lower() == target)
        )
        for _, meta in items:
            years = meta.get("years") if isinstance(meta, dict) else meta
            if years is not None:
                return True
    return False


def _repair_incomplete_skill_experience(filters: list[Filter]) -> list[Filter]:
    """Real, reproduced model failure, distinct from the one prompt.py's
    "guy with 3 years of exp in java" few-shot already targets: for some
    skill names -- confirmed reliably on "DevOps" across several phrasings
    ("2+ years of devops experience", "devops experience of 2+ years",
    "candidates with 2 years experience in devops") -- the LLM correctly
    picks the skill_experience FIELD (recognizing the years are tied to a
    named skill, not total career length) but leaves its `skill` sub-field
    empty, e.g. {"field":"skill_experience","operator":"gte","value":2}
    with no "skill" key at all -- while ALSO emitting a separate, redundant
    {"field":"skill","operator":"contains","value":"devops"} for the same
    skill. Without this repair, validation.py drops the incomplete
    skill_experience filter entirely (it has no skill to check against)
    and the years constraint is silently lost, leaving only the plain
    skill filter -- not wrong, but not what was asked either, and the
    generic drop message ("I need to know which skill...") is confusing
    when the recruiter typed the skill name right there in the same query.

    Repaired deterministically here rather than chased with yet another
    few-shot: this is the SECOND skill_experience emission shape confirmed
    broken on this model, and a code-level repair is trustworthy regardless
    of further model whims, per this session's own repeated lesson that
    prompt-only fixes on a 4B model don't reliably generalize.

    Only merges when there's exactly one incomplete skill_experience filter
    and exactly one plain skill filter to pair it with -- multiple of
    either is genuinely ambiguous (which skill goes with which threshold?)
    and is left alone, falling through to validation.py's existing safe,
    honest drop-with-explanation behavior."""
    incomplete = [
        i for i, f in enumerate(filters)
        if f.field == "skill_experience" and not f.skill
    ]
    plain_skill = [
        i for i, f in enumerate(filters)
        if f.field == "skill" and f.operator in {"contains", "equals"}
    ]
    if len(incomplete) != 1 or len(plain_skill) != 1:
        return filters

    exp_idx, skill_idx = incomplete[0], plain_skill[0]
    skill_value = str(filters[skill_idx].value)
    out = []
    for i, f in enumerate(filters):
        if i == skill_idx:
            continue
        if i == exp_idx:
            out.append(f.model_copy(update={"skill": skill_value}))
        else:
            out.append(f)
    return out


_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_SUBDOMAINS_DATASET_PATH = os.getenv(
    "SUBDOMAINS_DATASET_PATH", os.path.join(_ROOT, "master_subdomains_fixed.json")
)


@lru_cache(maxsize=1)
def _load_subdomain_names() -> dict[str, str]:
    """{normalized subdomain name: real-cased subdomain name}, from the SAME
    212-entry taxonomy the experience classifier uses to populate a real
    candidate's `domain` field (see experience_classifier.py) -- a curated
    list of real practice-area/position labels ("DevOps", "QA Testing",
    "Data Science", ...), a completely separate dataset from
    skill_taxonomy.py's tool/skill taxonomy. See
    _reclassify_skill_as_domain_when_its_a_position for why this matters:
    a query term matching one of THESE names but no real tool is a position
    a person works IN, not a skill they "have"."""
    if not os.path.isfile(_SUBDOMAINS_DATASET_PATH):
        return {}
    with open(_SUBDOMAINS_DATASET_PATH, "r", encoding="utf-8") as fh:
        entries = json.load(fh)
    return {
        entry["subdomain"].lower(): entry["subdomain"]
        for entry in entries if entry.get("subdomain")
    }


def _is_untracked_term(term: str | None) -> bool:
    """True when `term` is neither a known tool (skill_taxonomy's ~14,774
    canonical names) nor one of the 207 real subdomain/practice-area
    categories -- i.e. this dataset has NO tracked field a filter on `term`
    could ever match against, the same condition
    _reclassify_skill_as_domain_when_its_a_position already checks for its
    reclassify-vs-note decision. Shared here so
    _experience_text_matches can identify the exact same filters without
    duplicating the two lookups' meaning, just their (cheap) evaluation."""
    return bool(term) and not is_known_tool(term) and term.lower() not in _load_subdomain_names()


# Trailing generic head-noun words that name the THING being built/worked on
# rather than the actual domain concept -- "supply chain platform"'s real
# content is "supply chain"; "platform" only says it was software, which
# real resume prose essentially never repeats verbatim (confirmed live:
# real candidates' own text says "Supply Chain Optimization", "supply chain
# robustness", "Supply chain" -- NEVER literally "...platform", so requiring
# the recruiter's exact 3-word phrase found real matches in the dataset (see
# _experience_text_matches) but genuinely matched nobody until this existed).
# Same "strip the generic wrapper, keep the real concept" idea as
# skill_taxonomy._strip_skill_noise, just for domain phrases' trailing
# artifact-type noun instead of a skill's trailing meta-word.
_DOMAIN_NOISE_SUFFIXES = {
    "platform", "platforms", "system", "systems", "tool", "tools",
    "solution", "solutions", "software", "application", "applications",
    "product", "products", "service", "services", "framework", "portal",
}


def _strip_domain_noise_suffix(term: str) -> str | None:
    """Real, reported false-positive live bug: a 2-word phrase ("payment
    system") stripped down to a single bare common word ("payment") --
    every resume that ever mentions a payment ANYTHING (payment methods,
    PCI/"Payment Card Industry" audits, vendor payment issues, cash
    payment...) then counted as a "payment system" match, none of whom
    actually described building a payment system. Requiring the stripped
    result to still be 2+ words (>= 3 words before stripping) keeps the
    same "supply chain platform" -> "supply chain" win (a genuine phrase
    still identifiable on its own) while refusing to reduce a phrase down
    to a single word too generic to mean anything by itself -- same
    "a lone common word is not trustworthy evidence" principle as
    candidates._AMBIGUOUS_FOR_YEARS_TEXT_MATCH, just structural (word
    count) here instead of a curated word list, since the failure mode is
    inherent to stripping down to one word at all, not specific to which
    word it happens to be."""
    words = term.split()
    if len(words) >= 3 and words[-1].lower() in _DOMAIN_NOISE_SUFFIXES:
        return " ".join(words[:-1])
    return None


def _phrase_patterns(term: str) -> list:
    """[exact phrase, plural-tolerant variant] -- real, reported miss:
    Ahmed Sadig's own resume literally says "payment systems" (plural), but
    a recruiter typing "payment system" (singular) found nobody, because
    mention_pattern's `\\b...\\b` word boundary requires an exact word match
    and "systems" isn't a boundary-terminated occurrence of "system". Only
    appends the "+s" variant when `term` doesn't already end in "s" --
    doubling up ("analyticss") would be wrong for a term already plural."""
    patterns = [mention_pattern(term)]
    if not term.lower().endswith("s"):
        patterns.append(mention_pattern(term + "s"))
    return patterns


def _reclassify_skill_as_domain_when_its_a_position(
    filters: list[Filter],
) -> tuple[list[Filter], str | None]:
    """Real, reported gap: "3 years of exp in DevOps" got treated as a
    skill_experience filter with skill="DevOps" -- but DevOps isn't a tool
    anyone "has" (confirmed absent from the entire ~16,800-tool taxonomy,
    see skill_taxonomy.is_known_tool); it's a PRACTICE AREA/POSITION a
    person works IN, exactly matching one of the 212 real categories the
    experience classifier already uses to tag a candidate's ACTUAL work
    history (classified from their real job-description text, not a
    keyword list) into their `domain` field.

    A skill_experience filter (years attached) reclassifies to
    domain_experience -- NOT a bare `domain contains`, and the years are
    NOT dropped: real per-experience duration data exists and is summed per
    subdomain (see candidates._load_candidate_domain_years), so "2+ years
    of DevOps" means real years spent in DevOps-classified experience
    entries specifically, not total career length and not "ever did any
    DevOps work regardless of how long." A bare skill filter (no years)
    still reclassifies to plain `domain contains`, since there's no
    threshold to preserve.

    Only reclassifies when the term is NOT a real, recognized tool (a
    genuine tool name always wins -- this never overrides an actual skill
    match, e.g. "Docker" stays a skill) AND DOES exactly match a real
    subdomain label -- anything else (a genuine umbrella concept like
    "cloud" the taxonomy has no tool entry for either) is left untouched,
    same as before.

    A term that's NEITHER a known tool NOR a known subdomain (e.g. "supply
    chain platform" -- confirmed absent from both the ~14,774-tool taxonomy
    and all 207 real classified practice areas; unlike "DevOps", this
    dataset has NOTHING to widen or reclassify it into) still passes
    through unchanged, but a multi-word one also gets an honest note
    explaining it isn't tracked, rather than silently becoming an
    unexplained literal-phrase search that likely under-matches. Single-
    word misses stay silent -- a genuinely untracked ONE-word tool name is
    common and not worth an apologetic note.

    Real, reported gap in THIS note itself: it only ever fired for a term
    that started life as `skill`/`skill_experience` and got reclassified
    into domain -- a `domain`/`domain_experience` filter the LLM emitted
    directly (no reclassification needed, it's already the right field)
    got no equivalent check at all, so an untracked domain term like "B2B
    sales" (confirmed absent from every real subdomain, substring or
    otherwise) went completely quiet on `no_match` instead of explaining
    why. Handled below with the SAME multi-word-only honest-note rule,
    just checked as a SUBSTRING of any real subdomain (matching how
    `domain` filters actually get matched at apply time -- a candidate's
    real subdomain is often a longer, more specific string than the
    recruiter's search term, see _annotate_domain_match_years) rather than
    an exact match, since an exact-match miss here would wrongly flag a
    term like "sales operations" (a genuine substring of the real
    "Revenue Operations & Sales Operations" subdomain) as untracked.
    Never reclassifies these -- there is nothing to reclassify INTO, they
    already are the right field.

    Returns (filters, note) -- note explains the reclassification (or
    either untracked-term case above) for the response message, or None
    if nothing changed."""
    subdomains = _load_subdomain_names()
    if not subdomains:
        return filters, None

    out, notes = [], []
    for f in filters:
        if f.field in ("domain", "domain_experience"):
            term = f.skill if f.field == "domain_experience" else (
                f.value if f.operator in {"contains", "equals"} and isinstance(f.value, str) else None
            )
            out.append(f)
            if term and len(term.split()) >= 2 and not any(
                term.lower() in name for name in subdomains
            ):
                notes.append(
                    f'"{term}" isn\'t a specific skill or practice area this data tracks -- '
                    f"searching for that exact phrase literally, which may under-match."
                )
            continue

        term = None
        if f.field == "skill_experience":
            term = f.skill
        elif f.field == "skill" and f.operator in {"contains", "equals"}:
            term = f.value if isinstance(f.value, str) else None

        if term is None or is_known_tool(term):
            out.append(f)
            continue
        real_name = subdomains.get(term.lower())
        if real_name is None:
            out.append(f)
            # Real, reported gap: "supply chain platform" isn't a known
            # tool (checked -- absent from the ~14,774-canonical-name
            # taxonomy) AND isn't one of the 207 real classified practice
            # areas either (unlike "DevOps", which IS one) -- there is
            # NOTHING in this dataset to widen or reclassify it into, so it
            # silently became a literal, nobody-has-this-exact-phrase
            # search with no explanation. Multi-word-only (2+ words): a
            # single unrecognized tool name is common and NOT worth an
            # apologetic note (the taxonomy doesn't claim to know every
            # real tool) -- a multi-word phrase failing BOTH checks reads
            # much more like an untracked domain/product description, the
            # exact category this note is for.
            if len(term.split()) >= 2:
                notes.append(
                    f'"{term}" isn\'t a specific skill or practice area this data tracks -- '
                    f"searching for that exact phrase literally, which may under-match."
                )
            continue

        if f.field == "skill_experience":
            out.append(Filter(
                field="domain_experience", operator=f.operator, value=f.value,
                skill=real_name, hard=f.hard,
            ))
            notes.append(
                f'"{term}" is a practice area/position, not a specific skill -- matching '
                f'real years spent in "{real_name}"-classified experience instead of a skill.'
            )
        else:
            out.append(Filter(field="domain", operator="contains", value=real_name, hard=f.hard))
            notes.append(
                f'"{term}" is a practice area/position, not a specific skill -- matched '
                f"against real classified candidate experience instead."
            )
    return out, " ".join(notes) or None


_OR_WORD_RE = re.compile(r"\bor\b", re.IGNORECASE)


def _or_word_between(query: str, a: str, b: str) -> bool:
    """True only if BOTH `a` and `b` appear verbatim in `query` (case-
    insensitive) AND the word "or" sits between them in the text -- precise
    enough to tell a genuine "Kubernetes or Terraform" apart from some
    UNRELATED "or" elsewhere in a longer compound query (e.g. "...based in
    Mumbai or Delhi" must not also flip an unrelated skill pair to OR).
    False (never collapses) when either value isn't found verbatim in the
    query -- e.g. a canonicalized/aliased value that no longer matches the
    recruiter's literal wording -- since a safety net that can't verify its
    own trigger should stay quiet rather than guess, same principle as
    taxonomy._mentioned_in_query."""
    ql = query.lower()
    ia, ib = ql.find(a.lower()), ql.find(b.lower())
    if ia == -1 or ib == -1:
        return False
    start, end = sorted((ia, ib))
    return bool(_OR_WORD_RE.search(query[start:end]))


def _collapse_same_field_or_pairs(
    filters: list[Filter], query: str,
) -> tuple[list[Filter], str | None]:
    """Real, reported live bug: "Kubernetes or Terraform" (and, separately
    observed, "React or Angular", "Node.js or Django") is SUPPOSED to
    resolve to one `field in [...]` filter (see prompt.py rule 4) -- but on
    a compound query, this model sometimes instead emits TWO separate
    `contains` filters for the same field, each hard=True. Under AND logic
    that silently requires BOTH, the opposite of what "or" asked for --
    apply_spec has no way to recover the OR intent from two independently
    mandatory filters after the fact, and _fuzzy_skill_matches's own OR-
    widening never even applies (its skill_idx only looks at "in"/"not_in",
    since a genuine AND of two DIFFERENT required skills is not something
    that should ever be silently loosened).

    Deliberately gated on the query's own wording (see _or_word_between),
    not on the filter shape alone: two same-field `contains` filters is
    ALSO the exact shape of a genuinely-intended AND ("need someone who
    knows both Python and Django") -- the filters alone can't distinguish
    the two, only the query text says which one was meant, so this only
    fires when "or" is verifiably between the two specific values, never
    on filter shape or a stray "or" elsewhere in the query.

    Exact duplicate filters (same field/operator/value/hard -- a second,
    separately observed bug: "Kubernetes or Terraform" alone once produced
    FOUR filters, the same two values each listed twice) are always
    deduped first, unconditionally -- a literal duplicate can never
    represent a genuine second requirement, "or" or not."""
    deduped: list[Filter] = []
    seen = set()
    for f in filters:
        # Real, self-caught bug: omitting `f.skill` here silently dropped a
        # DIFFERENT skill_experience requirement -- "Kubernetes" and
        # "Terraform" skill_experience filters both happened to carry the
        # same `value=0` (an incomplete/unspecified-years marker), so
        # without `.skill` in the key they looked like exact duplicates of
        # EACH OTHER and only one survived, silently discarding Terraform's
        # requirement entirely. `f.skill` is the actual identity for
        # skill_experience/domain_experience filters, exactly as
        # Filter.key() already establishes (its own docstring: "a plain
        # field='skill' filter puts the skill name in `value` instead" --
        # `value` alone is NOT a reliable identity across every field).
        value_key = tuple(f.value) if isinstance(f.value, list) else f.value
        key = (f.field, f.operator, f.hard, value_key, f.skill)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)

    groups: dict[tuple[str, bool], list[int]] = {}
    for i, f in enumerate(deduped):
        if f.operator in ("contains", "equals") and isinstance(f.value, str):
            groups.setdefault((f.field, f.hard), []).append(i)

    to_collapse = {
        key: idxs for key, idxs in groups.items()
        if len(idxs) == 2
        and _or_word_between(query, deduped[idxs[0]].value, deduped[idxs[1]].value)
    }
    if not to_collapse:
        return deduped, None

    collapse_idx = {i for idxs in to_collapse.values() for i in idxs}
    out: list[Filter] = []
    notes = []
    handled = set()
    for i, f in enumerate(deduped):
        key = (f.field, f.hard)
        if i not in collapse_idx:
            out.append(f)
            continue
        if key in handled:
            continue
        handled.add(key)
        values = [deduped[j].value for j in to_collapse[key]]
        out.append(Filter(field=f.field, operator="in", value=values, hard=f.hard))
        notes.append(f'Read "{values[0]} or {values[1]}" as either one, not both required.')
    return out, " ".join(notes) or None


_NUMBER_WORD_RE = re.compile(
    r"\b(\d+(\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)


def _query_has_a_number(query: str) -> bool:
    return bool(_NUMBER_WORD_RE.search(query))


def _normalize_fabricated_skill_experience(
    filters: list[Filter], query: str,
) -> list[Filter]:
    """Real, reported live bug: a BARE "<skill> experience" mention with NO
    number stated anywhere ("Candidates with Python experience") is
    unreliably routed -- confirmed live, same session, same wording
    pattern: "Java experience"/"AWS experience" correctly became a plain
    `tools` entry (no threshold), but "Python experience"/"react
    experience" instead became `skill_experience(skill, gte, 1)` -- the
    model invents a threshold (always small: 0 or 1, confirmed both live)
    to satisfy the shape it decided to use, not because any number was
    actually said. A recruiter who typed zero numbers anywhere in their
    message cannot have stated a genuine per-skill years threshold, so
    ANY skill_experience/domain_experience filter surviving under that
    condition is, by construction, fabricated -- regardless of which
    small number it happened to land on (this subsumes the earlier,
    narrower "value<=0 is vacuous" check, which only caught the '0' half
    of the SAME underlying fabrication pattern; "1" is exactly as invented
    as "0", just harder to tell apart from a real "at least 1 year" ask by
    looking at the value alone).

    Deliberately gated on the QUERY's own content, not the value -- same
    grounding principle as taxonomy._mentioned_in_query and
    _or_word_between: a number genuinely stated ("Python experience of at
    least 2 years") must never be second-guessed, only a threshold that
    could not possibly have come from the query at all. Scoped to "the
    WHOLE query has no number anywhere" rather than trying to attribute a
    specific number to a specific skill -- safe and unambiguous when
    exactly one number-free skill_experience filter is present (the
    reported case); a query combining a genuinely-numbered field with a
    separate, differently-fabricated skill_experience for another skill is
    a rarer, harder-to-attribute case left alone rather than guessed at.

    Normalizing to a plain skill/domain filter HERE, before
    _collapse_same_field_or_pairs runs, means a redundant fabricated
    skill_experience(X) sitting next to a plain skill(X) filter (the
    Kubernetes-or-Terraform shape) is caught by that same-field dedup/
    OR-collapse like any other duplicate, instead of surviving as its own,
    differently-shaped mandatory requirement that silently reintroduces an
    AND (or, here, an invented minimum) the query never asked for."""
    out = []
    for f in filters:
        if (
            f.field in ("skill_experience", "domain_experience")
            and f.operator == "gte"
            and isinstance(f.value, (int, float))
            and (f.value <= 0 or not _query_has_a_number(query))
        ):
            new_field = "skill" if f.field == "skill_experience" else "domain"
            out.append(Filter(field=new_field, operator="contains", value=f.skill, hard=f.hard))
        else:
            out.append(f)
    return out


# Real, reported live bug: "high tire collage" (TWO typos at once --
# "tire" for "tier" AND "collage" for "college") made the model extract
# NOTHING at all (confirmed live: structured=[], tools=[], with no
# history at all) -- a fundamentally different, harder failure than the
# single-typo case _reclassify_typo_tier_filter fixes, since there is no
# filter object produced downstream to pattern-match and correct.
# "high tier college" (correctly spelled) resolves fine, confirming the
# model CAN handle the request -- it's specifically the compound
# misspelling that overwhelms it. Fixed further upstream instead: correct
# known typos in the QUERY TEXT itself before it ever reaches the LLM.
#
# "tire"/"tyre"/"teir"/"tir" have a real, unrelated, common meaning in
# ordinary English (the rubber wheel component) -- a blanket "tire" ->
# "tier" substitution would corrupt a legitimate query like "tire
# manufacturing experience" (a real automotive-industry search). Only
# corrected when immediately preceded by a tier qualifier (low/medium/
# high) -- the SAME contextual requirement _TIER_QUALIFIER_RE already
# uses for the post-resolution filter-value fix below -- since that
# specific combination has no plausible other meaning.
_TIRE_TYPO_QUERY_RE = re.compile(
    r"\b(low|medium|high)([\s-]+)(tires|tire|tyres|tyre|teir|tir)\b",
    re.IGNORECASE,
)

# These, unlike "tire", have no plausible unrelated meaning in an ordinary
# recruiting query, so a plain word-boundary substitution is safe without
# needing surrounding context.
_UNAMBIGUOUS_QUERY_TYPOS: dict[str, str] = {
    "collage": "college", "collages": "colleges",
    "colledge": "college", "colledges": "colleges",
    "univercity": "university", "unversity": "university",
    "univesity": "university", "universtiy": "university",
}
_UNAMBIGUOUS_QUERY_TYPO_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _UNAMBIGUOUS_QUERY_TYPOS) + r")\b",
    re.IGNORECASE,
)


def _correct_known_query_typos(query: str) -> str:
    """Corrects a small, curated set of confirmed-problematic misspellings
    in the raw query text BEFORE it reaches the LLM -- applied once, at the
    very top of _filter_by_query, so the corrected text flows through
    EVERYTHING downstream uniformly (the LLM call, taxonomy.py's query-
    grounding checks, and what gets stored in `history`): correcting the
    text sent to the model while leaving the original for grounding checks
    would make a NOW-correctly-resolved filter's raw_text mismatch the
    (still uncorrected) query it's checked against, wrongly tripping
    taxonomy._mentioned_in_query's "this doesn't appear in the message"
    guard on a filter that's actually fine.

    Deliberately a curated substitution dictionary, not open-ended fuzzy/
    edit-distance matching, and deliberately narrow in scope (a handful of
    CONFIRMED problem words, not an attempt at general spell-checking):
    this codebase already learned the hard way (see
    skill_taxonomy._fuzzy_typo_match's 10-character floor) that loose
    fuzzy matching on short words collides with too many unrelated real
    words by chance -- "tier"/"college" are exactly that short. The
    frontend echoes the user's own typed message immediately and
    independently of this (see search-ui/index.html) -- correcting here
    only affects what the LLM/backend sees, never what the recruiter sees
    themselves having typed."""
    query = _TIRE_TYPO_QUERY_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}tier", query)
    return _UNAMBIGUOUS_QUERY_TYPO_RE.sub(
        lambda m: _UNAMBIGUOUS_QUERY_TYPOS[m.group(0).lower()], query,
    )


_NARROWING_PHRASE_RE = re.compile(
    r"\b(out of (all|these|those|them)|from (these|those|them|the current results)|"
    r"among (these|those|them)|of (the )?(above|these|those))\b",
    re.IGNORECASE,
)


def _query_signals_narrowing(query: str) -> bool:
    """True when the query's own phrasing ("out of all", "from these",
    "among those", "of the above", ...) means the recruiter wants to filter
    FURTHER within the already-active result set, not start over.

    Real, reported live bug: turn 1 set a Python filter; turn 2, "out of
    all give me guy from high tire comapany", made the model emit
    replace_all=true -- reading "out of all" as a reset instruction rather
    than what it plainly is in this phrasing, a narrowing qualifier over
    the existing results. Downstream that silently offered to DROP the
    active Python filter (it shows on the confirm screen, but defaulted
    UNCHECKED, exactly the kind of thing a recruiter clicking "Search"
    without reading closely would lose without meaning to) instead of
    combining it with the new company-tier filter as the recruiter's own
    words asked for. Overriding replace_all to False whenever the query
    itself uses this phrasing is safe even with nothing active to
    preserve (spec.filters empty) -- it's simply a no-op then."""
    return bool(_NARROWING_PHRASE_RE.search(query))


# Contact-detail fields this dataset never tracks at all (not "hard to
# answer", genuinely absent from every candidate record) -- confirmed live:
# "what's his email address" resolved to intent LOOKUP with
# lookup_field="email", which _answer_lookup then had no way to honestly
# decline (it only knows how to report ALLOWED_FIELDS values, "email" isn't
# one). Caught here, deterministically, before _answer_lookup ever runs --
# regardless of what the model's own intent/lookup_field said -- since the
# model has repeatedly been observed to route a contact-detail question as
# LOOKUP anyway despite prompt.py's own worked example for this exact case.
_UNTRACKED_LOOKUP_RE = re.compile(
    r"\b(email|e-mail|phone( number)?|contact( details?| info(rmation)?)?"
    r"|linkedin|resume|r[ée]sum[ée]|cv|github|portfolio)\b",
    re.IGNORECASE,
)


def _is_untracked_lookup(query: str) -> bool:
    """True when `query` asks about a contact/identity detail this dataset
    has no field for at all (email, phone, contact info, LinkedIn, resume/
    CV, GitHub, portfolio) -- as opposed to a real, tracked field (college,
    notice period, work history, ...) that never matches any of these
    specific terms, e.g. "which college did he go to" or "what's her
    notice period" are untouched."""
    return bool(_UNTRACKED_LOOKUP_RE.search(query))


# A REQUIRED qualifier word immediately before it makes the overall 2-word
# phrase distinctive enough to reclassify safely -- but "tier" alone is
# only 4 letters, too short for open-ended edit-distance fuzzy matching
# (see skill_taxonomy._fuzzy_typo_match's own 10-character floor, raised
# for exactly this reason: a short word collides with too many unrelated
# entries by pure chance). A small curated list of realistic typos/
# misspellings is safer and sufficient here.
_TIER_QUALIFIER_RE = re.compile(
    r"^\s*(low|medium|high)[\s-]+(tier|tiers|tire|tires|tyre|tyres|teir|tir)\s*$",
    re.IGNORECASE,
)

# The two string fields that each pair with a real ordinal tier field --
# reclassification target for _reclassify_typo_tier_filter.
_TIER_FIELD_MAP = {"company": "company_tier", "university": "college_tier"}

# Real, reported live bug: "give me guy with python exprence in high tire
# company" (a query the pre-LLM typo pass already corrects to "...high tier
# company") STILL sometimes landed the model on `{"field":"skill","value":
# "high tier"}` -- not company/university at all, so _TIER_FIELD_MAP's
# lookup above never fires. No real tool/skill is ever named "high tier"/
# "low tier"/etc (the pattern below is fully anchored, so it can never
# collide with a genuine skill value), so a `skill` filter matching it is
# unambiguously this same misreading wearing a different field -- just
# with no company/university on the FILTER itself to say which ordinal
# tier field it meant. Disambiguated from the query's own wording instead:
# a college/university/school mention nearby means college_tier, otherwise
# company_tier (the overwhelmingly common phrasing in practice -- every
# real report of this bug this session was about company tier).
_COLLEGE_CONTEXT_RE = re.compile(
    r"\b(college|university|school|campus|alma\s*mater)\b", re.IGNORECASE,
)


def _reclassify_typo_tier_filter(
    filters: list[Filter], query: str,
) -> tuple[list[Filter], str | None]:
    """Real, reported live bug: "high tire company" ("tier" mistyped as
    "tire") wasn't recognized as a company-tier request at all -- the
    model fell back to treating "high tire" as if it were itself a
    literal COMPANY NAME to search for (`company contains "high tire"`),
    which can never match anyone (no company is named "high tire").
    Deterministic pattern match on the filter's OWN value, not a prompt
    fix (this session's repeated lesson: prompt-only fixes don't reliably
    generalize on this model): a `company`/`university` value that's
    exactly "<low/medium/high> <a tier typo>" (see _TIER_QUALIFIER_RE) is
    reclassified into the real `company_tier`/`college_tier` filter
    instead, `gte` (the standard operator for these two ordinal fields --
    see the "high tier company" -> `company_tier gte "High"` precedent),
    same qualifier casing candidates._load_company_ranks_data already
    stores ("Low"/"Medium"/"High"). Also separately catches the same
    misreading landing on a bare `skill` filter instead (see
    _COLLEGE_CONTEXT_RE above for how the ambiguous case is resolved)."""
    out = []
    notes = []
    for f in filters:
        tier_field = _TIER_FIELD_MAP.get(f.field)
        ambiguous_skill = False
        if tier_field is None and f.field == "skill" and f.operator in {"contains", "equals"}:
            tier_field = "college_tier" if _COLLEGE_CONTEXT_RE.search(query) else "company_tier"
            ambiguous_skill = True
        m = tier_field and isinstance(f.value, str) and _TIER_QUALIFIER_RE.match(f.value)
        if not m:
            out.append(f)
            continue
        qualifier = m.group(1).title()
        out.append(Filter(field=tier_field, operator="gte", value=qualifier, hard=f.hard))
        if ambiguous_skill:
            what = "college" if tier_field == "college_tier" else "company"
            notes.append(
                f'Read "{f.value}" as "{qualifier} tier" -- searching {tier_field} '
                f"({what}) instead of a literal skill named that, which doesn't exist."
            )
        else:
            what = "company" if f.field == "company" else "university"
            notes.append(
                f'Read "{f.value}" as "{qualifier} tier" -- searching {tier_field} '
                f"instead of a literal {what} name, which nobody would ever match."
            )
    return out, " ".join(notes) or None


def _repair_resolved_filters(
    filters: list[Filter], query: str,
) -> tuple[list[Filter], str | None]:
    """The shared repair tail applied to any already-resolved filter list --
    both the top-level list (whichever of v1's expand_skill_filters or v2's
    taxonomy.resolve_filters produced it) and, unchanged, each
    AlternativeGroup's filters (which are ALWAYS v1-shaped/already-resolved
    by construction, see AlternativeGroup's docstring, so they only need
    this tail, not a resolution pass of their own). Runs the same five
    deterministic fixups in the same order either way: normalize a
    fabricated (query-ungrounded, or logically-vacuous "gte 0")
    skill_experience/domain_experience into a plain skill/domain filter,
    reclassify a typo'd tier phrase out of company/university into the
    real ordinal tier field, pair an incomplete skill_experience filter
    with its stray plain-skill sibling, reclassify a position/practice-area
    term out of skill into domain, then canonicalize any country filter's
    spelling."""
    filters = _normalize_fabricated_skill_experience(filters, query)
    filters, tier_note = _reclassify_typo_tier_filter(filters, query)
    filters = _repair_incomplete_skill_experience(filters)
    filters, domain_note = _reclassify_skill_as_domain_when_its_a_position(filters)
    filters = [_canonicalize_country_filter(f) for f in filters]
    note = " ".join(m for m in (tier_note, domain_note) if m) or None
    return filters, note


def _expand_seniority_filters(
    filters: list[Filter], degrade_to_years_only: bool,
) -> tuple[list[Filter], list[AlternativeGroup], str | None]:
    """A bare level word ("fresher"/"mid level"/"senior"/"lead") resolves to
    a DEFINED years range plus a job_title keyword check, rather than the
    generic "ask the recruiter for a number every time" CLARIFY path (see
    prompt.py/prompt_v2.py's vague-threshold rule) -- once a band has a real,
    agreed-on definition (vocabulary.SENIORITY_BANDS), it's no longer an
    undefined guess. The LLM's only job is recognizing WHICH band was named
    (a classification task it handles well) and emitting a trivial
    {"field":"seniority","operator":"equals","value":"<term>"} marker; ALL
    the real logic -- canonicalizing the term, building the years range,
    building the title check, combining them as OR -- happens here,
    deterministically. This mirrors this session's own hard-learned lesson:
    the newly-built alternative_groups mechanism was found live to be
    UNRELIABLE when the LLM had to construct an OR-of-routes itself on a
    complex query -- this feature deliberately never asks it to.

    A candidate qualifies for a band if EITHER their total years fall in
    range OR their job title matches one of the band's keywords -- most real
    titles don't literally state a level word, so requiring both would
    under-match badly. That's expressed as two separate AlternativeGroups
    (OR'd against each other and against everything else, unchanged
    apply_spec/merge_alternative_groups machinery) -- NOT as two flat
    Filters, because AlternativeGroup.filters bypasses merge_filters'
    Filter.key()-based dedup entirely; a bounded band's two flat experience
    filters (gte + lte) would otherwise collide on the same merge key
    (`experience` is not in _MULTI_VALUE_FIELDS) and merge_filters'
    last-write-wins would silently drop one of the two bounds.

    Every "seniority" filter is removed from the returned flat list either
    way -- it must never reach validate_filters/apply_spec directly, neither
    of which know what to do with it (no real candidate record has a literal
    "seniority" key).

    `degrade_to_years_only`: FilterSpec.alternative_groups is one flat,
    non-nested OR-level (see AlternativeGroup's own docstring) and cannot
    express "(OR-set A) AND (OR-set B)" -- so when this turn (or an earlier,
    still-active turn) already has a REAL, unrelated alternative_groups
    statement, injecting a second OR-group here would either wrongly flatten
    two independent either/or requirements into one (satisfying either alone
    passes both), or -- via merge_alternative_groups' wholesale-replace
    semantics -- silently discard the earlier one. When this is True, a
    recognized HARD band degrades to a single flat `experience gte min`
    filter (floor only, same shape and same reason as the SOFT case below --
    see that comment) instead of the full years-range-OR-title-match, with a
    note explaining why the ceiling was dropped -- a looser-than-intended
    but honestly-labeled match beats silently discarding or misapplying an
    unrelated real either/or requirement.
    """
    keep: list[Filter] = []
    new_groups: list[AlternativeGroup] = []
    notes: list[str] = []
    for f in filters:
        if f.field != "seniority":
            keep.append(f)
            continue
        band_key = seniority_band(f.value if isinstance(f.value, str) else None)
        if band_key is None:
            notes.append(
                f'I don\'t recognize "{f.value}" as a seniority level -- '
                f"try fresher, junior, mid, senior, or lead."
            )
            continue
        band = SENIORITY_BANDS[band_key]
        if not f.hard:
            # A soft "prefer mid-level" preference: floor-only, deliberately
            # -- a bounded band's SECOND bound can't safely be a second flat
            # `experience` filter here (Filter.key() doesn't distinguish by
            # operator, so `experience` isn't in _MULTI_VALUE_FIELDS --
            # merge_filters would collapse two same-field filters down to
            # whichever is processed last, silently dropping the other, even
            # within the SAME incoming list -- confirmed via
            # test_merge_still_replaces_single_value_fields). Never excludes
            # anyone regardless (schema_v2's hard=false contract); it only
            # nudges ranking via _apply_soft_preferences.
            keep.append(Filter(field="experience", operator="gte",
                                value=band["min"], hard=False))
            continue
        if degrade_to_years_only:
            keep.append(Filter(field="experience", operator="gte", value=band["min"], hard=True))
            notes.append(
                f'Applying "{band_key}" as at least {band["min"]} years -- not narrowed to '
                f"the full range or job title this turn, since another either/or "
                f"requirement is already active and the two can't be combined in one search."
            )
            continue
        years_filters = [Filter(field="experience", operator="gte", value=band["min"], hard=True)]
        if band["max"] is not None:
            years_filters.append(Filter(field="experience", operator="lte", value=band["max"], hard=True))
        new_groups.append(AlternativeGroup(filters=years_filters))
        new_groups.append(AlternativeGroup(filters=[
            Filter(field="job_title", operator="in", value=list(band["title_keywords"]), hard=True),
        ]))
    return keep, new_groups, " ".join(notes) or None


def _domain_skill_options(job_id: str, domain_term: str, limit: int = 8) -> list[str]:
    """Real, data-grounded skill choices for a vague domain-only query ("I
    want a DevOps guy") -- intersects the curated tool taxonomy's list for
    that practice area (see skill_taxonomy.tools_for_subdomain, ~188 real
    entries for "DevOps") with skills that ACTUALLY appear among this job's
    real candidates classified into it, ranked by how many candidates have
    each one. Never offers a tool nobody in this pool actually has -- the
    full taxonomy list can run to ~200 tools, an overwhelming and mostly-
    irrelevant-here list to hand a recruiter. Empty list (caller falls back
    to searching the domain alone) if the taxonomy has nothing for this
    label, or if none of its tools happen to appear among this job's real
    domain-matching candidates."""
    taxonomy_tools = {t.lower(): t for t in tools_for_subdomain(domain_term)}
    if not taxonomy_tools:
        return []

    counts: dict[str, int] = {}
    for c in get_matched_candidates(job_id):
        domain_years = c.get("domain_years") or {}
        if not any(domain_term.lower() in k.lower() for k in domain_years):
            continue
        for s in {s.lower() for s in skill_names_of(c)}:
            if s in taxonomy_tools:
                counts[s] = counts.get(s, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [taxonomy_tools[k] for k, _ in ranked[:limit]]


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
        """Thin wrapper around `_filter_by_query` so every return path
        below -- pending lookup/combine/clarify, each LLM intent,
        validation failure, no_match, ok -- gets persisted to the query log
        exactly once, in one place, instead of instrumenting each return
        individually. See query_log.py for what's actually kept (the raw
        LLM JSON alongside the translated/repaired filters, status, counts
        -- not the full candidate PII list).

        `debug_info` is a fresh dict per call (not instance state -- `self`
        is a module-level singleton shared across concurrent requests, see
        app/api/main.py, so anything per-request must live on the stack,
        never on `self`), filled in by `_filter_by_query` as it runs so the
        raw LLM output reaches the log without threading a new return value
        through every one of that method's many return statements."""
        debug_info: dict = {}
        response = self._filter_by_query(query, job_id, session_id, reset, debug_info=debug_info)
        log_query(
            session_id=session_id, job_id=job_id, query=query, response=response,
            raw_llm_output=debug_info.get("llm_out"),
        )
        return response

    def _filter_by_query(
        self, query: str, job_id: str, session_id: str, reset: bool = False,
        debug_info: dict | None = None,
    ) -> FilterResponse:
        # Applied ONCE, here, before anything else touches `query` -- see
        # _correct_known_query_typos's docstring for why the correction
        # must flow through the LLM call, grounding checks, and stored
        # history uniformly rather than only some of them.
        query = _correct_known_query_typos(query)
        if reset:
            self.store.clear(session_id, job_id)

        current = self.store.get(session_id, job_id)
        spec = current.spec
        logger.info(
            "REQUEST_IN job_id=%s session_id=%s reset=%s query=%r "
            "active_filters=%s pending_clarify=%s pending_confirm=%s pending_lookup_field=%s",
            job_id, session_id, reset, query,
            [f.model_dump(exclude_none=True) for f in spec.filters],
            bool(current.pending_clarify), bool(current.pending_confirm),
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

        # Same idea for a pending CONFIRM ("here are your active filters --
        # keep, drop, or add each one"): a bare "yes"/"no" resolves
        # deterministically, no LLM call, applying every default_checked
        # filter verbatim rather than re-deriving anything (see
        # PendingConfirm's docstring). The real, expected path is the
        # recruiter clicking checkboxes in the UI and submitting straight to
        # the PATCH endpoint (see patch_state) -- this fast-path exists only
        # so typing "yes"/"no" in the chat box, which every other pending
        # state in this app supports, still works here too.
        if current.pending_confirm and len(query.split()) <= 6:
            answer = _extract_yes_no(query)
            if answer is True:
                pc = current.pending_confirm
                kept = [c.filter for c in pc.choices if c.default_checked]
                return self._validate_apply_persist(
                    kept, pc.logic, job_id, session_id, query,
                    current.history, extra_message=pc.message,
                    alternative_groups=pc.alternative_groups,
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
            # confirm and fall through to a fresh query below.

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
        if debug_info is not None:
            debug_info["llm_out"] = llm_out.model_dump(exclude_none=True)

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
            if _is_untracked_lookup(query):
                message = "This data doesn't track contact details (email, phone, LinkedIn, etc.)."
                self.store.set(session_id, job_id, SessionState(
                    spec=spec, last_candidates=current.last_candidates,
                    history=_append_history(current.history, query, message),
                ))
                return FilterResponse(
                    status="unsupported", message=message,
                    logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
                )
            return self._answer_lookup(
                llm_out, current, spec, job_id, session_id, query,
            )

        if llm_out.intent == "EXPERIENCE_SEARCH":
            # Real, reported live bug: "backend engineers who worked on a
            # supply chain platform" (a compound ask naming BOTH a real
            # structured requirement AND an achievement, in the SAME turn)
            # silently dropped "backend engineers" entirely and searched the
            # achievement phrase alone, matching non-engineers too --
            # `llm_out.filters` was never merged into the spec this call
            # actually narrows the pool with, only a filter set ALREADY
            # active from an earlier turn was. Merge this turn's own
            # `filters` in first, exactly like FILTER_CANDIDATES would,
            # before narrowing/searching -- `llm_out.filters` here is
            # already v1-resolved shape regardless of prompt schema (v2's
            # EXPERIENCE_SEARCH doesn't populate structured/tools).
            experience_spec = spec
            if llm_out.filters:
                merged_filters = (
                    llm_out.filters if llm_out.replace_all
                    else merge_filters(spec.filters, llm_out.filters)
                )
                experience_spec = FilterSpec(
                    logic=spec.logic, filters=merged_filters,
                    alternative_groups=spec.alternative_groups,
                )
            return self._answer_experience_search(
                llm_out, current, experience_spec, job_id, session_id, query,
            )

        # FILTER_CANDIDATES. Two ways to arrive at a resolved filter list,
        # depending on which schema produced llm_out:
        #   v1 -- the LLM already emitted final field/operator/value
        #     filters; expand_skill_filters() widens any umbrella skill
        #     concept against the curated taxonomy.
        #   v2 -- the LLM only reported raw spans/buckets (structured/
        #     tools); app/core/taxonomy.py resolves them into the same
        #     Filter shape, reusing the identical taxonomy lookups.
        # Either way, resolve any colloquial country name ("USA", "UK") to
        # the exact spelling candidates are tagged with uniformly, as a
        # POST-pass over whichever list resulted -- a fixed lookup table,
        # not something worth asking the LLM to memorize (see
        # candidates.canonicalize_country), and not something taxonomy.py
        # duplicates internally.
        # See _query_signals_narrowing's docstring: the model's own
        # replace_all flag is overridden to False whenever the query's
        # phrasing ("out of all", "from these", ...) says the recruiter
        # means to narrow the active results further, not reset them --
        # used below in place of llm_out.replace_all everywhere it would
        # otherwise decide whether prior filters/groups survive.
        replace_all = llm_out.replace_all and not _query_signals_narrowing(query)

        skip_notes: list[str] = []
        if getattr(self.llm, "prompt_schema", "v1") == "v2":
            resolved_filters, skip_notes = taxonomy.resolve_filters(
                llm_out, query, taxonomy.active_filter_terms(spec.filters),
            )
        else:
            resolved_filters = expand_skill_filters(llm_out.filters)
        expanded_filters, domain_reclass_note = _repair_resolved_filters(resolved_filters, query)
        expanded_filters, or_collapse_note = _collapse_same_field_or_pairs(expanded_filters, query)

        # A real, unrelated alternative_groups statement -- this turn's own,
        # or an earlier turn's still-active one that would otherwise survive
        # -- means a seniority band's own OR-of-routes can't safely be
        # expanded into a SECOND set of alternative_groups: FilterSpec.
        # alternative_groups is one flat, non-nested OR-level (see
        # AlternativeGroup's docstring) and cannot express "(OR-set A) AND
        # (OR-set B)" -- see _expand_seniority_filters' own docstring for the
        # full reasoning, including the merge_alternative_groups wholesale-
        # replace hazard for the carried-over case.
        degrade_seniority_to_years_only = bool(llm_out.alternative_groups) or (
            bool(spec.alternative_groups) and not replace_all
        )
        expanded_filters, seniority_groups, seniority_note = _expand_seniority_filters(
            expanded_filters, degrade_seniority_to_years_only,
        )

        # Each AlternativeGroup's filters are ALREADY v1-shaped/resolved by
        # construction (see AlternativeGroup's docstring) regardless of
        # whether the main query used v1 or v2 -- they only need the same
        # repair tail, not a taxonomy/expand_skill_filters resolution pass.
        repaired_groups: list[AlternativeGroup] = []
        group_reclass_notes: list[str] = []
        for g in llm_out.alternative_groups:
            repaired, note = _repair_resolved_filters(g.filters, query)
            repaired_groups.append(AlternativeGroup(filters=repaired))
            if note:
                group_reclass_notes.append(note)
        repaired_groups.extend(seniority_groups)

        extra_message = " ".join(
            m for m in (llm_out.message, *skip_notes, domain_reclass_note,
                        or_collapse_note, seniority_note, *group_reclass_notes) if m
        ) or None
        effective = (
            expanded_filters if replace_all
            else merge_filters(spec.filters, expanded_filters)
        )
        effective_groups = merge_alternative_groups(
            spec.alternative_groups, repaired_groups, replace_all,
        )

        # A query that names a practice area/position but no specific tool
        # ("I want a DevOps guy") would otherwise just mean "anyone ever
        # classified into this domain" -- the same vagueness complaint that
        # led to domain_experience existing at all. Offer real skill
        # choices to narrow it first, rather than searching immediately.
        # Checked against THIS turn's own new filters (expanded_filters),
        # not the merged `effective` -- "also devops" on top of an existing
        # Python filter is still vague about DevOps specifically, even
        # though a skill IS active overall. Skipped entirely (falls through
        # to a normal search) if the taxonomy has no tools for this label,
        # or none of them appear among this job's real matching candidates
        # -- see _domain_skill_options; asking with an empty list would be
        # a broken, useless prompt.
        domain_filters_this_turn = [
            f for f in expanded_filters if f.field in ("domain", "domain_experience")
        ]
        if domain_filters_this_turn and not any(f.field == "skill" for f in expanded_filters):
            domain_filter = domain_filters_this_turn[0]
            term = domain_filter.value if domain_filter.field == "domain" else domain_filter.skill
            skill_options = _domain_skill_options(job_id, term) if isinstance(term, str) else []
            if skill_options:
                message = " ".join(m for m in (
                    extra_message,
                    f'Which specific skills matter for this "{term}" search? '
                    f"Check any that apply, or hit Search with none checked to see everyone.",
                ) if m)
                self.store.set(session_id, job_id, SessionState(
                    spec=spec, last_candidates=current.last_candidates,
                    pending_domain_skill_pick=PendingDomainSkillPick(
                        domain_filter=domain_filter, skill_options=skill_options, message=message,
                    ),
                    history=_append_history(current.history, query, message),
                ))
                return FilterResponse(
                    status="domain_skill_pick", message=message,
                    domain_filter=domain_filter, skill_options=skill_options,
                    logic=spec.logic, filters=spec.filters, chips=to_chips(spec.filters),
                )

        # A query that DROPS a field on top of an already-active search
        # shows every filter as its own row for the recruiter to keep or
        # drop individually -- see PendingConfirm's docstring for why
        # granularity (not smarter guessing about when to ask) is the
        # actual fix here: a bundled "do you want X and Y?" question can
        # hide a stale filter the recruiter agrees to without noticing.
        # Checked by KEY (field+value for multi-value fields, field+skill
        # otherwise -- see Filter.key()), not bare field name, so e.g. two
        # different skill filters are never collapsed into "the same field."
        #
        # Real, reported live complaint: confirm used to fire on EVERY
        # change to the active key set, additions included -- "also needs
        # AWS experience" on top of an existing Python filter is completely
        # unambiguous (recruiter said "also"; nothing is at risk of being
        # silently lost, merge_filters is purely additive/overwriting by
        # key and can only DROP an existing key when replace_all=True says
        # so explicitly) but still stopped the recruiter with a checklist
        # for something that should have just searched immediately. A pure
        # addition carries no ambiguity worth a confirm step -- the new chip
        # is right there afterward, one click from removal if it's wrong,
        # exactly like every other filter. Confirm is now reserved for the
        # one case that's actually risky: something ACTIVE would silently
        # disappear (only possible under replace_all).
        #
        # A query that only UPDATES an already-active field's VALUE (same
        # key set, e.g. "actually Mumbai instead of Bangalore") is likewise
        # unambiguous -- nothing about WHICH filters are active is in
        # question there -- so it still auto-applies with no confirm step.
        existing_keys = {f.key() for f in spec.filters}
        effective_keys = {f.key() for f in effective}
        dropped_keys = existing_keys - effective_keys
        if spec.filters and dropped_keys:
            choices: list[FilterChoice] = []
            # A field dropped entirely (only possible under replace_all)
            # still gets its own row so the recruiter can consciously
            # re-add it -- default UNCHECKED, since replace_all said drop
            # it, but visible and one click away from surviving if that
            # was wrong.
            for f in spec.filters:
                if f.key() not in effective_keys:
                    choices.append(FilterChoice(
                        filter=f, label=chip_label(f),
                        origin="existing", default_checked=False,
                    ))
            for f in effective:
                origin = "existing" if f.key() in existing_keys else "new"
                choices.append(FilterChoice(
                    filter=f, label=chip_label(f), origin=origin, default_checked=True,
                ))
            prompt = "Review your filters below and hit Search."
            message = " ".join(m for m in (extra_message, prompt) if m)
            self.store.set(session_id, job_id, SessionState(
                spec=spec, last_candidates=current.last_candidates,
                pending_confirm=PendingConfirm(
                    choices=choices, logic=llm_out.logic, message=message,
                    alternative_groups=effective_groups,
                ),
                history=_append_history(current.history, query, prompt),
            ))
            return FilterResponse(
                status="confirm", choices=choices, message=message,
                logic=llm_out.logic, filters=spec.filters,
                alternative_groups=spec.alternative_groups,
                chips=to_chips(spec.filters, spec.alternative_groups),
            )

        return self._validate_apply_persist(
            effective, llm_out.logic, job_id, session_id, query, current.history,
            alternative_groups=effective_groups,
            extra_message=extra_message,
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

        if not experience_index.index_exists():
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
        hard_spec = _hard_only(spec)
        pool = apply_spec(candidates, hard_spec) if hard_spec.filters else candidates
        pool_ids = {c.get("id") for c in pool}
        by_id = {c.get("id"): c for c in pool}

        # Restrict-then-rank, not rank-then-restrict: `search_candidates`
        # scores every chunk against ONLY this pool before cutting to top_k,
        # so a real match can't be pushed out by closer-scoring chunks from
        # candidates entirely outside this job/filter set. A prior version
        # of this call used `search(query, top_k=200)` (rank the WHOLE
        # corpus first, filter to the pool after) -- that hasn't been
        # observed to misfire on the current dataset/threshold (checked
        # against pools as small as one candidate), but it's a latent
        # failure mode with no upside, not a deliberate tradeoff, so it's
        # fixed here rather than left as a "probably fine for now."
        try:
            hits = experience_index.search_candidates(
                llm_out.experience_query, pool_ids, top_k=len(pool_ids),
            )
        except Exception:
            logger.warning(
                "experience_index.search_candidates failed for %r",
                llm_out.experience_query, exc_info=True,
            )
            hits = []

        # Below _EXPERIENCE_MIN_SIMILARITY (see that constant's docstring
        # for the empirical basis) a "match" is noise, not a genuine hit.
        # `hits` is SUPPOSED to already be one row per candidate (their
        # best-scoring experience), restricted to `pool_ids` -- that's
        # `search_candidates`'s contract, not something this loop can
        # itself verify -- so both parts of it are still defended here
        # rather than trusted blindly: a hit for an id outside the pool is
        # skipped (guarded with `.get()`), and a second hit for a candidate
        # already seen keeps only the higher score (`best_score`) instead
        # of adding a duplicate row. Neither should happen with a real
        # `search_candidates` call; both are one-line insurance against
        # that guarantee ever regressing, not active logic under normal use.
        best_score: dict[str, float] = {}
        for hit in hits:
            score = float(hit.get("score", 0.0))
            if score < self._EXPERIENCE_MIN_SIMILARITY:
                continue
            cid = hit.get("candidate_id")
            if cid not in by_id:
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

        chips = to_chips(spec.filters) + [
            Chip(label=f'\U0001f50e "{llm_out.experience_query}"', field="experience_query"),
        ]
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
            req.filters, req.logic, req.job_id, req.session_id,
            alternative_groups=req.alternative_groups,
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
    ) -> FilterResponse:
        available = get_available_fields(job_id)
        result = validate_filters(filters, available)
        validated_groups, group_notes = validate_alternative_groups(
            alternative_groups or [], available,
        )
        extra_message = " ".join(m for m in (extra_message, *group_notes) if m) or None
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
                alternative_groups=current.spec.alternative_groups,
                chips=to_chips(current.spec.filters, current.spec.alternative_groups),
            )

        spec = FilterSpec(logic=logic, filters=result.filters, alternative_groups=validated_groups)
        candidates = get_matched_candidates(job_id)

        # skill_experience needs a per-skill YEARS number (see
        # engine.extract_value) -- candidates._skill_years_from_experience
        # now computes a real one for known tools where the resume's own
        # text supports it, but NOT for every skill: an unrecognized skill
        # name, or a real tool nobody in THIS job's pool happened to name in
        # their own job-description prose, still has nothing. Checked PER
        # skill_experience filter's own named skill (never pool-wide -- see
        # _skill_years_available's docstring for why that would silently
        # reintroduce a false no_match), so one filter can resolve normally
        # while a sibling filter for a different, genuinely-undated skill
        # still degrades to a plain "has this skill" filter (still gets the
        # normal fuzzy/related-tool widening below) with an explanation.
        skill_exp_idx = [i for i, f in enumerate(spec.filters) if f.field == "skill_experience"]
        undated = [i for i in skill_exp_idx if not _skill_years_available(candidates, spec.filters[i].skill)]
        if undated:
            converted = list(spec.filters)
            skills_named = []
            for i in undated:
                f = converted[i]
                skills_named.append(f.skill)
                converted[i] = Filter(field="skill", operator="contains", value=f.skill, hard=f.hard)
            spec = FilterSpec(logic=spec.logic, filters=converted,
                               alternative_groups=spec.alternative_groups)
            names = " / ".join(dict.fromkeys(skills_named))
            note = (
                f"This data doesn't reliably track years spent on {names} specifically "
                f"(only total career years, plus real per-skill years where a resume's own "
                f"job description happens to name the tool) -- showing everyone with "
                f"{names} instead."
            )
            extra_message = " ".join(m for m in (extra_message, note) if m)

        # `spec` (hard+soft) is what gets persisted/rendered as chips below
        # -- only the actual inclusion/exclusion decision uses the
        # hard-only view, so a soft (schema_v2 "nice to have") filter never
        # excludes anyone, per Filter.hard's contract.
        hard_spec = _hard_only(spec)
        filtered = apply_spec(candidates, hard_spec)
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
            job_id, hard_spec, matched_ids={c.get("id") for c in filtered},
        )
        # Same widening, reaching into alternative_groups branches too (see
        # _fuzzy_alternative_group_matches) -- a separate pass because a
        # route has no partial-credit tier, only full-match-or-not.
        full_extra = full_extra + self._fuzzy_alternative_group_matches(
            job_id, hard_spec,
            matched_ids={c.get("id") for c in filtered} | {c.get("id") for c in full_extra},
        )
        # Widen an untracked domain/skill term (no known tool, no known
        # subdomain -- see _is_untracked_term) by searching real job-history
        # text for a literal mention of it, instead of only checking a
        # tracked field the term was never going to be found in (see
        # _experience_text_matches). Stays completely silent when nothing
        # is found, same as before this existed.
        text_extra, text_terms_found = self._experience_text_matches(
            job_id, hard_spec,
            matched_ids={c.get("id") for c in filtered} | {c.get("id") for c in full_extra},
        )
        # Real, reported live bug: a candidate missing ONLY an untracked
        # term (e.g. "cloud technologies") sat in `partial` from
        # _fuzzy_skill_matches (which has nothing in the taxonomy to ever
        # confirm an untracked term, so it can only ever report it
        # "missing") -- but the SAME candidate could ALSO independently
        # turn up in `text_extra` above, since real job-history text DOES
        # confirm it. Without reconciling the two, they appeared TWICE in
        # the final list: once tagged partial ("missing: cloud
        # technologies"), once as an untagged full match. Promote instead
        # of merely deduping -- the text search already confirmed they
        # satisfy the exact thing partial was calling missing, so they
        # belong in full_extra, not still flagged as incomplete.
        text_extra_ids = {c.get("id") for c in text_extra}
        partial = [c for c in partial if c.get("id") not in text_extra_ids]
        full_extra = full_extra + text_extra
        if text_terms_found:
            names = " / ".join(text_terms_found)
            extra_message = " ".join(m for m in (
                extra_message,
                f'Found by searching real job-history text for "{names}" -- not a tracked '
                f"category, so this may miss people who did this but never wrote it that way.",
            ) if m)
        # Strict TIER order, never interleaved by raw match_score across
        # tiers -- a real, reported bug: a fuzzy/related-tool match (e.g.
        # "Jira" satisfied via a related tool, not the literal word) could
        # outrank a genuine 100%-exact match just because its own,
        # unrelated match_score happened to be higher (confirmed live:
        # "Project Managers: Agile + Jira, 10+ years" on job 00000103 put a
        # single related-tool match FIRST, ahead of 6 real exact matches).
        # Tier 1: exact (every requirement literally satisfied). Tier 2:
        # full_extra -- satisfies EVERY requirement, but at least one only
        # via a curated related-tool widening (see fuzzy_skill_match) --
        # still a complete match, just not a literal one, so it ranks below
        # exact but above partial. Tier 3: partial -- missing at least one
        # requirement entirely (see partial_skill_match), always last.
        # match_score still orders candidates WITHIN each tier.
        filtered.sort(key=lambda c: c.get("match_score", 0), reverse=True)
        if full_extra:
            full_extra.sort(key=lambda c: c.get("match_score", 0), reverse=True)
            filtered = filtered + full_extra
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

        # Soft preferences (schema_v2's hard=false, see Filter.hard) never
        # excluded anyone above -- hard_spec already dropped them before
        # apply_spec ran. Here they finally do something: survivors of the
        # hard filters get RE-RANKED by how many soft preferences they also
        # satisfy, not filtered further. AND-only, same restriction as
        # _hard_only (see its docstring for why OR/NOT have no coherent
        # hard/soft split).
        soft_filters = [f for f in spec.filters if not f.hard] if spec.logic == "AND" else []
        if soft_filters:
            filtered = self._apply_soft_preferences(filtered, soft_filters)

        # Real, reported UX gap: a domain/domain_experience query ("who has
        # worked in fintech") showed the candidate's TOTAL career length on
        # their card (e.g. 10 years) even when they'd spent only a fraction
        # of that in the actually-queried domain (e.g. 4 years in FinTech
        # specifically) -- the real per-experience duration data already
        # exists (see candidates._load_candidate_domain_years), it just
        # wasn't surfaced here the way it already was for domain_experience
        # filtering itself. Tags each matching candidate with
        # `domain_match_years` for card-level display, alongside (not
        # replacing) their real total `experience`.
        filtered = _annotate_domain_match_years(filtered, spec)
        # Same real-per-experience-duration idea, for skills -- see
        # candidates._skill_years_from_experience: a "python guy" search
        # otherwise showed the candidate's TOTAL career length on their
        # card, easily misread as years of Python specifically (real,
        # reported confusion). Tags each candidate with `skill_match_years`
        # when a real number exists for a searched skill; silently absent
        # (not zero) when it doesn't, same as the underlying data.
        filtered = _annotate_skill_match_years(filtered, spec)

        # Persist the new valid state, including who's now in view -- this
        # is what a later LOOKUP ("which college did he go to") resolves
        # against.
        assistant_summary = (
            f"Applied filters: {', '.join(c.label for c in to_chips(spec.filters, spec.alternative_groups))}"
            if spec.filters or spec.alternative_groups else "Cleared all filters"
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
                chips=to_chips(spec.filters, spec.alternative_groups),
                message=skip_note or "No candidates match these filters.",
                suggestions=_no_match_suggestions(spec.filters),
            )

        return FilterResponse(
            status="ok",
            total=len(candidates),
            showing=len(filtered),
            logic=spec.logic,
            filters=spec.filters,
            alternative_groups=spec.alternative_groups,
            chips=to_chips(spec.filters, spec.alternative_groups),
            candidates=filtered,
            message=skip_note,
        )

    # ------------------------------------------------------------------ #
    # Widen skill filters so a candidate counts as a match via a curated
    # related tool, not just the exact word -- merged directly into the
    # real result.
    #
    # An LLM-verified semantic-similarity tier used to sit here too (an
    # embedding shortlist + an LLM judging whether it counts). REMOVED --
    # confirmed live it does not work on this project's model/hardware: with
    # thinking off (needed for latency), qwen3:4b rubber-stamped EVERY
    # candidate as a match regardless of the skill asked, including one
    # whose only listed skills were "cooking, painting, yoga, gardening"
    # against a "Java" query. With thinking on, it reasoned correctly but a
    # realistic shortlist (up to 8 candidates) took 15+ minutes and still
    # didn't finish -- not viable for an interactive search. Rather than
    # ship a knob with no setting that is both correct and fast, this tier
    # is off: only the two deterministic checks below (exact word, curated
    # related-tool taxonomy) decide a skill match now. See app/core/
    # semantic.py and app/llm/skill_verify.py -- kept, not deleted, as a
    # real, working building block if a viable verification approach
    # (a faster model, a non-LLM classifier, etc.) is found later.
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
    def _fuzzy_skill_matches(
        self, job_id: str, spec: FilterSpec, matched_ids: set,
    ) -> tuple[list[dict], list[dict]]:
        """Returns (full_extra, partial) -- full_extra: candidates who now
        satisfy the ENTIRE spec. partial: candidates who satisfy every
        non-skill filter plus SOME (not all) skill filters under AND logic --
        each dict carries a `partial_skill_match` key: {"matched": int,
        "total": int, "missing": [label, ...]} for card-level display.

        Neither list silently passes off a fuzzy hit as if it were the exact
        word the recruiter typed: any candidate (full_extra or partial) whose
        qualification for a skill filter came from a curated related-tool
        relation -- not the literal skill name -- carries a
        `fuzzy_skill_match` key: [{"skill": label, "matched_via": "related"}
        , ...] so the recruiter can see and judge it, same principle as
        `partial_skill_match` but for the case that was previously
        indistinguishable from a real exact match.

        Also covers a same-field alternative among named tools ("Kubernetes
        or Terraform" -> `skill in ["Kubernetes","Terraform"]`, see
        prompt.py rule 4) -- previously that `in`/`not_in` filter was
        matched by apply_spec via pure literal string equality only (see
        engine._op_in), so a resume saying "K8s" wouldn't satisfy a filter
        asking for "Kubernetes" even though the identical alias already
        widens a plain `contains` filter. A candidate counts as satisfying
        an `in` filter if they qualify (exact OR related) for AT LEAST ONE
        of the listed values, mirroring `_op_in`'s own "any of these
        values" semantics.

        Deliberately out of scope: `spec.alternative_groups` (see
        schemas.AlternativeGroup). A skill filter living inside a group is
        matched literally by apply_spec's group-gate (matches_filter,
        exact/contains only) and gets NEITHER related-tool widening NOR
        partial-match surfacing here -- same "explicitly scope out for
        reliability/complexity, don't silently under-support" precedent as
        this project's removed LLM-verification semantic tier. See
        `_fuzzy_alternative_group_matches` below, which covers that case
        separately (full-match only, no partial-credit tier)."""
        skill_idx = [
            i for i, f in enumerate(spec.filters)
            if f.field == "skill" and f.operator in {"contains", "not_contains", "in", "not_in"}
        ]
        if not skill_idx:
            return [], []

        candidates = get_matched_candidates(job_id)
        pool = [c for c in candidates if c.get("id") not in matched_ids]
        if not pool:
            return [], []

        # Per skill filter: the set of candidate ids (from `pool`) that
        # qualify via a curated taxonomy relation -- deterministic, no
        # network/LLM call. `match_kind_by_filter[i][cid]` records the
        # provenance ("related") so it can be surfaced later -- an "exact"
        # hit (the literal word, or any listed value for "in"/"not_in") needs
        # no provenance tag at all.
        qualifying_by_filter: dict[int, set] = {}
        match_kind_by_filter: dict[int, dict[str, str]] = {}
        for i in skill_idx:
            f = spec.filters[i]
            values = f.value if isinstance(f.value, list) else [f.value]
            canon_values = [canonicalize(str(v)) for v in values]

            qualifies = set()
            kinds: dict[str, str] = {}
            for c in pool:
                cand_skills = {s.lower() for s in skill_names_of(c)}
                exact_hit = False
                related_hit = False
                for canon in canon_values:
                    exact, related = related_terms_for(canon)
                    if cand_skills & exact:
                        exact_hit = True
                        break
                    if cand_skills & related:
                        related_hit = True
                if exact_hit or related_hit:
                    qualifies.add(c.get("id"))
                    if not exact_hit:
                        kinds[c.get("id")] = "related"

            qualifying_by_filter[i] = qualifies
            match_kind_by_filter[i] = kinds

        # Real, reported live bug: a filter's own value list can contain
        # several DIFFERENT-looking entries that all canonicalize to the
        # SAME name (confirmed live: an expand_skill_term("machine
        # learning") call made before this session's alias-bloat fix
        # returned the canonical name plus 17 of its own near-duplicate
        # aliases -- "ML model", "ML pipeline", "predictive modeling", etc.
        # -- all of which canonicalize() resolves right back to "machine
        # learning"). Joining every value unconditionally rendered a
        # fuzzy_skill_match badge as "machine learning/machine learning/...
        # (18x)/Scikit Learn/PyTorch/..." -- `dict.fromkeys` dedupes the
        # CANONICALIZED form while preserving first-seen order, so a label
        # reads as one clean skill name per distinct concept regardless of
        # how many raw aliases happen to be sitting in the value list.
        skill_labels = {
            i: "/".join(dict.fromkeys(canonicalize(str(v)) for v in (
                spec.filters[i].value if isinstance(spec.filters[i].value, list)
                else [spec.filters[i].value]
            )))
            for i in skill_idx
        }

        full_extra, partial = [], []
        for c in pool:
            cid = c.get("id")
            non_skill_ok = True
            skill_hits, skill_misses = [], []
            fuzzy_hits: list[dict] = []
            for i, f in enumerate(spec.filters):
                if i in qualifying_by_filter:
                    hit = cid in qualifying_by_filter[i]
                    if f.operator in {"not_contains", "not_in"}:
                        hit = not hit
                    (skill_hits if hit else skill_misses).append(skill_labels[i])
                    kind = match_kind_by_filter[i].get(cid)
                    if hit and f.operator not in {"not_contains", "not_in"} and kind:
                        fuzzy_hits.append({"skill": skill_labels[i], "matched_via": kind})
                else:
                    non_skill_ok = non_skill_ok and matches_filter(c, f)

            if spec.logic == "OR":
                if non_skill_ok or skill_hits:
                    enriched = dict(c) if fuzzy_hits else c
                    if fuzzy_hits:
                        enriched["fuzzy_skill_match"] = fuzzy_hits
                    full_extra.append(enriched)
            elif spec.logic == "NOT":
                if not non_skill_ok and not skill_hits:
                    full_extra.append(c)
            else:  # AND
                if not non_skill_ok:
                    continue
                if not skill_misses:
                    enriched = dict(c) if fuzzy_hits else c
                    if fuzzy_hits:
                        enriched["fuzzy_skill_match"] = fuzzy_hits
                    full_extra.append(enriched)
                elif skill_hits:
                    enriched = dict(c)
                    enriched["partial_skill_match"] = {
                        "matched": len(skill_hits),
                        "total": len(skill_hits) + len(skill_misses),
                        "missing": skill_misses,
                    }
                    if fuzzy_hits:
                        enriched["fuzzy_skill_match"] = fuzzy_hits
                    partial.append(enriched)
        return full_extra, partial

    # ------------------------------------------------------------------ #
    # Same curated-related-tool widening as _fuzzy_skill_matches, but
    # reaching into `spec.alternative_groups` branches -- a real, previously
    # documented gap: a skill filter living inside a route (e.g. "either a
    # PhD or hands-on Kubernetes experience") was matched by apply_spec's
    # group-gate via `matches_filter` only, i.e. literal exact/contains, no
    # alias widening at all, even though the identical alias already widens
    # the equivalent TOP-LEVEL filter.
    #
    # Deliberately narrower than _fuzzy_skill_matches: no partial-credit
    # tier here -- a route is "AND of a route's own filters, OR across
    # routes", and there is no established UI concept for "partially
    # satisfied one route" the way there is for "matched some but not all
    # AND'd skill filters" -- so a route only ever counts as satisfied or
    # not. `spec.filters`/`spec.logic` (the flat AND/OR/NOT gate) still must
    # hold strictly, exactly as apply_spec requires -- only the group-gate
    # itself is widened.
    # ------------------------------------------------------------------ #
    def _fuzzy_alternative_group_matches(
        self, job_id: str, spec: FilterSpec, matched_ids: set,
    ) -> list[dict]:
        if not spec.alternative_groups:
            return []
        candidates = get_matched_candidates(job_id)
        pool = [c for c in candidates if c.get("id") not in matched_ids]
        if not pool:
            return []

        def flat_gate_ok(c: dict) -> bool:
            if not spec.filters:
                return True
            checks = [matches_filter(c, f) for f in spec.filters]
            if spec.logic == "OR":
                return any(checks)
            if spec.logic == "NOT":
                return not any(checks)
            return all(checks)

        def branch_satisfied(c: dict, cand_skills: set, branch_filters: list[Filter]):
            fuzzy: list[dict] = []
            for f in branch_filters:
                if f.field == "skill" and f.operator in {"contains", "not_contains", "in", "not_in"}:
                    values = f.value if isinstance(f.value, list) else [f.value]
                    exact_hit, related_label = False, None
                    for v in values:
                        canon = canonicalize(str(v))
                        exact, related = related_terms_for(canon)
                        if cand_skills & exact:
                            exact_hit = True
                            break
                        if related_label is None and cand_skills & related:
                            related_label = canon
                    hit = exact_hit or related_label is not None
                    if f.operator in {"not_contains", "not_in"}:
                        hit = not hit
                        related_label = None
                    if not hit:
                        return False, []
                    if related_label:
                        fuzzy.append({"skill": related_label, "matched_via": "related"})
                elif not matches_filter(c, f):
                    return False, []
            return True, fuzzy

        full_extra = []
        for c in pool:
            if not flat_gate_ok(c):
                continue
            cand_skills = {s.lower() for s in skill_names_of(c)}
            fuzzy_hits: list[dict] | None = None
            for g in spec.alternative_groups:
                if not g.filters:
                    continue
                ok, fuzzy = branch_satisfied(c, cand_skills, g.filters)
                if ok:
                    fuzzy_hits = fuzzy
                    break
            if fuzzy_hits is None:
                continue
            enriched = dict(c) if fuzzy_hits else c
            if fuzzy_hits:
                enriched["fuzzy_skill_match"] = fuzzy_hits
            full_extra.append(enriched)
        return full_extra

    # ------------------------------------------------------------------ #
    # Real, reported gap: a domain/skill term this dataset has NO tracked
    # field for at all (e.g. "supply chain platform" -- absent from both the
    # ~14,774-tool taxonomy and the 207 real subdomain categories, see
    # _reclassify_skill_as_domain_when_its_a_position) previously became a
    # silent, almost-certainly-empty literal-field search with nothing to
    # widen it into. This does the same deterministic keyword search
    # candidates._skill_years_from_experience already does for a
    # candidate's OWN declared, taxonomy-known skills -- but over an
    # arbitrary untracked phrase, searching every job's real position +
    # description text (see candidates._adapt_resume's `experience_text`)
    # for a literal mention of it.
    #
    # Full-match only, same as _fuzzy_alternative_group_matches -- no
    # partial-credit tier here either, and deliberately AND-only (see
    # flat_gate_ok below): OR/NOT's "how many of several loosely-defined
    # text mentions count" has no obviously correct semantics the way a
    # strict AND does, so it's left unhandled rather than guessed at.
    # ------------------------------------------------------------------ #
    def _experience_text_matches(
        self, job_id: str, spec: FilterSpec, matched_ids: set,
    ) -> tuple[list[dict], list[str]]:
        """Returns (full_extra, terms_found) -- terms_found lists each
        distinct untracked term that actually matched at least one
        candidate this way, for the caller to fold into a response note
        (silent, same as before, when nothing is found -- see
        test_untracked_multiword_phrase_gets_an_honest_note_not_silence,
        which relies on staying silent in exactly that case)."""
        if spec.logic != "AND":
            return [], []

        def term_of(f: Filter) -> str | None:
            if f.field in ("domain", "skill") and f.operator in ("contains", "equals"):
                return f.value if isinstance(f.value, str) else None
            if f.field in ("domain_experience", "skill_experience"):
                return f.skill
            return None

        idx = [
            i for i, f in enumerate(spec.filters)
            # Multi-word only -- real, reported live bug: a compound query
            # caused the model to also emit a spurious `skill contains
            # "backend"` filter (a bare, common English word, not a real
            # tool -- already covered by the query's own `job_title
            # contains "backend engineer"`). An unrestricted single-word
            # text search would have free-text-matched "backend" against
            # nearly any engineering resume, the exact same false-positive
            # class as the "payment"-from-"payment system" bug, just
            # arriving via LLM over-extraction instead of noise-suffix
            # stripping. Same "a lone common word isn't trustworthy
            # evidence" principle as _strip_domain_noise_suffix and
            # candidates._AMBIGUOUS_FOR_YEARS_TEXT_MATCH -- and the same
            # multi-word floor _reclassify_skill_as_domain_when_its_a_position
            # already uses for its own honest note, now applied to the
            # widening itself, not just the message.
            if term_of(f) and len(term_of(f).split()) >= 2
            and _is_untracked_term(term_of(f))
        ]
        if not idx:
            return [], []

        candidates = get_matched_candidates(job_id)
        pool = [c for c in candidates if c.get("id") not in matched_ids]
        if not pool:
            return [], []

        terms = {i: term_of(spec.filters[i]) for i in idx}
        # Try the exact phrase (singular + plural, see _phrase_patterns)
        # first; if it also has a generic trailing artifact-type noun
        # ("platform", "system", ...) AND stripping it still leaves 2+
        # words, also accept just the real concept in front of it -- see
        # _strip_domain_noise_suffix's docstring for why a 1-word result
        # is deliberately never attempted.
        patterns = {
            i: _phrase_patterns(terms[i]) + (
                _phrase_patterns(core)
                if (core := _strip_domain_noise_suffix(terms[i])) else []
            )
            for i in idx
        }
        other_filters = [f for i, f in enumerate(spec.filters) if i not in idx]

        texts = experience_texts_by_candidate()
        full_extra = []
        terms_found: set[str] = set()
        for c in pool:
            text = texts.get(c.get("id"))
            if not text:
                continue
            if not all(any(p.search(text) for p in patterns[i]) for i in idx):
                continue
            if not all(matches_filter(c, f) for f in other_filters):
                continue
            enriched = dict(c)
            enriched["experience_text_match"] = [{"term": terms[i]} for i in idx]
            full_extra.append(enriched)
            terms_found.update(terms[i] for i in idx)
        return full_extra, sorted(terms_found)

    # ------------------------------------------------------------------ #
    # Soft preferences (schema_v2's hard=false, see Filter.hard and
    # _hard_only): re-rank hard-filter survivors by how many soft
    # preferences they ALSO satisfy, rather than excluding non-matches.
    # Uses engine.matches_filter directly -- no engine.py changes needed,
    # and deliberately no fuzzy/taxonomy widening for a soft SKILL filter
    # (that machinery exists to decide who counts as a HARD match; a soft
    # preference is scored as a strict yes/no against the stored data,
    # same as any other soft field).
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_soft_preferences(
        candidates: list[dict], soft_filters: list[Filter],
    ) -> list[dict]:
        soft_labels = [c.label for c in to_chips(soft_filters)]
        scored = []
        for c in candidates:
            hits = [i for i, f in enumerate(soft_filters) if matches_filter(c, f)]
            enriched = dict(c)
            enriched["soft_match"] = {
                "satisfied": [soft_labels[i] for i in hits],
                "missing": [soft_labels[i] for i in range(len(soft_filters)) if i not in hits],
                "count": len(hits),
            }
            scored.append(enriched)
        # Stable sort: within the same soft-match count, the order
        # apply_spec/fuzzy-matching already produced (real match_score,
        # then fuzzy-extra, then partial) is preserved -- soft preferences
        # break ties among equals, they don't override the primary ranking.
        scored.sort(key=lambda c: -c["soft_match"]["count"])
        return scored


def _annotate_domain_match_years(filtered: list[dict], spec: FilterSpec) -> list[dict]:
    """Tags each candidate with `domain_match_years`: [{"domain": real
    subdomain name, "years": N}, ...] for every one of their REAL subdomains
    (see candidates._load_candidate_domain_years) that matches a `domain`
    or `domain_experience` filter in `spec` -- a candidate can genuinely
    have more than one (e.g. "FinTech" AND "Payments & FinTech Engineering"
    both containing "fintech"). `domain` filters match by SUBSTRING (same
    semantics as engine.py's own domain matching, since a candidate's real
    subdomain is often a longer, more specific string than the recruiter's
    search term); `domain_experience` filters already carry the exact real
    subdomain name in `.skill`.

    Also scans `spec.alternative_groups`' filters (a route can legitimately
    contain a domain leaf) -- display-only, so a missed annotation here
    would be a cosmetic gap, not a matching-correctness bug.

    Returns a NEW list -- copies a candidate dict only when it actually
    gets a tag, never mutates in place. `get_matched_candidates` is
    `@lru_cache`d and shared across every request; mutating one of its
    dicts here would leak this one query's annotation into every future
    request that happens to return the same candidate."""
    all_filters = list(spec.filters) + [
        f for g in spec.alternative_groups for f in g.filters
    ]
    domain_terms: list[str] = []
    for f in all_filters:
        if f.field == "domain" and isinstance(f.value, str):
            domain_terms.append(f.value.lower())
        elif f.field == "domain_experience" and f.skill:
            domain_terms.append(f.skill.lower())
    if not domain_terms:
        return filtered

    out = []
    for c in filtered:
        domain_years = c.get("domain_years") or {}
        matches = [
            {"domain": subdomain, "years": years}
            for subdomain, years in domain_years.items()
            if any(term in subdomain.lower() for term in domain_terms)
        ]
        if matches:
            c = dict(c)
            c["domain_match_years"] = matches
        out.append(c)
    return out


def _annotate_skill_match_years(filtered: list[dict], spec: FilterSpec) -> list[dict]:
    """Tags each candidate with `skill_match_years`: [{"skill": real skill
    name, "years": N}, ...] for every `skill`/`skill_experience` filter in
    `spec` where candidates._skill_years_from_experience computed a REAL
    number for THIS candidate -- so the card can show "3.6 yrs in Python"
    instead of only the total-career-years line, which is easily (and was
    actually, per real reported feedback) misread as being specific to
    whatever skill the search was for. Mirrors _annotate_domain_match_years's
    shape/scoping decisions.

    Most candidates still won't have a real number for a given skill (see
    _skill_years_from_experience's docstring on real, uneven per-tool prose-
    mention rates) -- silently omitted for those rather than showing
    anything in its place; the card's existing total-years line already
    covers them.

    Also scans `spec.alternative_groups`' filters (a route can legitimately
    contain a skill leaf) -- display-only, so a missed annotation here is
    cosmetic, not a matching-correctness bug.

    Returns a NEW list, same non-mutation discipline as
    _annotate_domain_match_years (get_matched_candidates is lru_cache'd and
    shared across requests)."""
    all_filters = list(spec.filters) + [
        f for g in spec.alternative_groups for f in g.filters
    ]
    skill_terms: list[str] = []
    for f in all_filters:
        if f.field == "skill":
            values = f.value if isinstance(f.value, list) else [f.value]
            skill_terms.extend(str(v) for v in values)
        elif f.field == "skill_experience" and f.skill:
            skill_terms.append(f.skill)
    if not skill_terms:
        return filtered
    targets = {t.lower() for t in skill_terms}

    out = []
    for c in filtered:
        skills = c.get("skills")
        matches = []
        if isinstance(skills, dict):
            for name, meta in skills.items():
                if name.lower() not in targets:
                    continue
                years = meta.get("years") if isinstance(meta, dict) else meta
                if years is not None:
                    matches.append({"skill": name, "years": years})
        if matches:
            c = dict(c)
            c["skill_match_years"] = matches
        out.append(c)
    return out


def _no_match_suggestions(filters: list[Filter]) -> list[str]:
    tips = ["Remove one of the filters", "Search all locations"]
    for f in filters:
        if f.field == "skill_experience":
            tips.append(f"Reduce the {f.skill} experience requirement")
        if f.field == "domain_experience":
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
