"""Eval harness for the NL -> filter translation layer.

WHY THIS EXISTS
---------------
Every test in `tests/` uses a FakeLLM: they prove the pipeline AROUND the
model is correct, and prove nothing at all about the model's actual
translation quality. Model behaviour has so far only ever been checked
anecdotally -- one query at a time, minutes per query, findings recorded in
prose in SESSION_NOTES.md. That is not a basis for deciding whether a prompt
change, a model swap, or a fine-tune helped.

This module scores a set of labelled cases (eval/corpus.jsonl) against
whatever produced predictions for them, and reports the numbers that
actually matter for this system's stated goal ("never silently wrong").

DELIBERATELY INTERFACE-TOLERANT
-------------------------------
Expectations are written SEMANTICALLY ("there should be a location=Mumbai
equals predicate", "it should ask about experience", "it should report
industry as untracked") rather than as literal current-schema JSON. The
LLM interface is expected to change (tool-calls, span tagging, a nested
query tree); the question "what is the correct answer to this recruiter
utterance" does not change with it. Anything that reduces to a set of
predicates plus a verdict can be scored here without rewriting the corpus.

The corpus doubles as the seed retrieval/training bank -- same labelled
pairs, three uses (score today, retrieve few-shots, fine-tune later).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable, Optional

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.jsonl")

# Intents that mean "the system declined to act and said so" -- the safe
# answers. Predicting a filter when one of these was correct is the
# dangerous failure this whole design is meant to prevent, so it gets its
# own headline metric rather than being averaged into general accuracy.
_ABSTAIN_INTENTS = {"CLARIFY", "UNSUPPORTED_FILTER"}


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    id: str
    query: str
    expect: dict
    history: list[dict] = dc_field(default_factory=list)
    current_filters: list[dict] = dc_field(default_factory=list)
    tags: list[str] = dc_field(default_factory=list)
    note: str = ""

    @property
    def expected_intent(self) -> str:
        return self.expect["intent"]


def load_corpus(path: str | None = None) -> list[Case]:
    """Read corpus.jsonl. One JSON object per line; blank lines and lines
    starting with '#' are skipped so the file can carry section headers."""
    path = path or _CORPUS_PATH
    cases: list[Case] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            case = Case(
                id=raw["id"], query=raw["query"], expect=raw["expect"],
                history=raw.get("history", []),
                current_filters=raw.get("current_filters", []),
                tags=raw.get("tags", []), note=raw.get("note", ""),
            )
            if case.id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return cases


# --------------------------------------------------------------------------- #
# Value normalisation -- so 5 vs 5.0, "Mumbai" vs "mumbai", and list order
# never count as disagreements. Only real semantic differences should score
# as errors; a scorer that punishes formatting produces numbers nobody
# trusts and therefore nobody uses.
# --------------------------------------------------------------------------- #
def _norm_scalar(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return str(v).strip().lower()


def _norm_value(v: Any) -> Any:
    if isinstance(v, list):
        return sorted((str(_norm_scalar(x)) for x in v))
    return _norm_scalar(v)


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def predicate_matches(expected: dict, actual: dict) -> bool:
    """Does one actual predicate satisfy one expected predicate spec?

    `value_includes` (instead of `value`) asserts only that the listed terms
    are present in the actual value. Needed for umbrella-concept expansions
    ("machine learning" -> a list of concrete tools): which specific tools
    the model proposes is genuinely open, but the concept term itself must
    be there (prompt rule 3 requires it first in the array), and the
    taxonomy expands the rest afterwards anyway. Asserting an exact list
    would score correct answers as wrong.
    """
    if expected["field"] != actual.get("field"):
        return False
    if expected["operator"] != actual.get("operator"):
        return False

    if "skill" in expected:
        if _norm_scalar(expected["skill"]) != _norm_scalar(actual.get("skill") or ""):
            return False
    if "unit" in expected:
        if _norm_scalar(expected["unit"]) != _norm_scalar(actual.get("unit") or ""):
            return False

    if "value_includes" in expected:
        have = {str(x) for x in _as_list(_norm_value(actual.get("value")))}
        want = {str(_norm_scalar(x)) for x in expected["value_includes"]}
        return want.issubset(have)
    if "value" in expected:
        return _norm_value(expected["value"]) == _norm_value(actual.get("value"))
    return True


def match_predicates(
    expected: list[dict], actual: list[dict]
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Greedy bijection between expected and actual predicates.

    Returns (matched_pairs, missing, extra). `extra` is the important one:
    an actual predicate matching nothing expected is a fabricated filter --
    prompt rule 6a-i ("every filter must trace to a word actually in the
    query") violated, and a documented real failure mode of this system
    under compound load.
    """
    remaining = list(actual)
    matched: list[tuple[dict, dict]] = []
    missing: list[dict] = []
    for exp in expected:
        hit = next((a for a in remaining if predicate_matches(exp, a)), None)
        if hit is None:
            missing.append(exp)
        else:
            remaining.remove(hit)
            matched.append((exp, hit))
    return matched, missing, remaining


# --------------------------------------------------------------------------- #
# Per-case scoring
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    case: Case
    predicted: dict
    intent_ok: bool
    matched: list[tuple[dict, dict]]
    missing: list[dict]
    extra: list[dict]
    metadata_ok: bool
    metadata_problems: list[str]
    error: Optional[str] = None

    @property
    def exact(self) -> bool:
        """Fully correct: right verdict, right filters, nothing invented,
        right resolution metadata."""
        return (
            self.error is None and self.intent_ok and not self.missing
            and not self.extra and self.metadata_ok
        )

    @property
    def silent_wrong(self) -> bool:
        """Emitted filters when the correct answer was to ask or to decline.
        The single most damaging failure mode: the recruiter gets a
        confident, plausible, wrong result set with no indication anything
        was guessed."""
        return (
            self.case.expected_intent in _ABSTAIN_INTENTS
            and self.predicted.get("intent") == "FILTER_CANDIDATES"
            and bool(self.predicted.get("filters"))
        )

    @property
    def over_clarify(self) -> bool:
        """Asked a question when the query was already explicit. Annoying,
        not dangerous -- tracked separately so it can't be traded off
        against silent_wrong by accident."""
        return (
            self.case.expected_intent == "FILTER_CANDIDATES"
            and self.predicted.get("intent") == "CLARIFY"
        )

    @property
    def fabricated(self) -> bool:
        return bool(self.extra)


def _check_metadata(expect: dict, predicted: dict) -> tuple[bool, list[str]]:
    """Check the resolution metadata a correct answer must carry.

    These are not cosmetic. clarify_field/clarify_operator are what let the
    backend turn the recruiter's next short reply into a real filter without
    another LLM call; clarify_value is what lets a bare "yes" resolve in
    code. A CLARIFY that omits them is a question the system cannot act on
    the answer to -- so it is scored as wrong even though the question text
    may read fine.
    """
    problems: list[str] = []
    for key in ("clarify_field", "clarify_operator", "clarify_value",
                "clarify_skill", "clarify_unit", "lookup_field"):
        if key not in expect:
            continue
        want, got = expect[key], predicted.get(key)
        if want is None:
            if got is not None:
                problems.append(f"{key}: expected unset, got {got!r}")
        elif got is None or _norm_scalar(want) != _norm_scalar(got):
            problems.append(f"{key}: expected {want!r}, got {got!r}")

    for needle in expect.get("message_mentions", []):
        msg = (predicted.get("message") or "")
        if needle.lower() not in msg.lower():
            problems.append(f"message missing {needle!r}")

    for banned in expect.get("forbid_fields", []):
        if any(f.get("field") == banned for f in predicted.get("filters", [])):
            problems.append(f"emitted a forbidden {banned!r} filter")

    if "logic" in expect:
        want_logic = str(expect["logic"]).strip().upper()
        got_logic = str(predicted.get("logic", "AND")).strip().upper()
        if want_logic != got_logic:
            problems.append(f"logic: expected {want_logic!r}, got {got_logic!r}")

    return (not problems), problems


def score_case(case: Case, predicted: dict, error: str | None = None) -> CaseResult:
    """`predicted` is a plain dict: {intent, filters: [...], clarify_field,
    lookup_field, message, ...} -- deliberately not an LLMOutput, so a future
    tool-call interface can be scored by mapping its output into this shape
    instead of rewriting the corpus."""
    predicted = predicted or {}
    intent_ok = predicted.get("intent") == case.expected_intent
    expected_preds = case.expect.get("predicates", [])
    actual_preds = predicted.get("filters", []) or []

    # Only compare filters when the correct answer has filters. For a
    # CLARIFY/UNSUPPORTED case, any emitted filter is already counted by
    # silent_wrong; also calling each one "extra" would double-penalise the
    # same mistake and distort precision.
    if expected_preds or case.expected_intent == "FILTER_CANDIDATES":
        matched, missing, extra = match_predicates(expected_preds, actual_preds)
    else:
        matched, missing, extra = [], [], []

    metadata_ok, problems = _check_metadata(case.expect, predicted)
    return CaseResult(
        case=case, predicted=predicted, intent_ok=intent_ok, matched=matched,
        missing=missing, extra=extra, metadata_ok=metadata_ok,
        metadata_problems=problems, error=error,
    )


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
def _rate(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def summarise(results: Iterable[CaseResult]) -> dict:
    results = list(results)
    n = len(results)
    tp = sum(len(r.matched) for r in results)
    fn = sum(len(r.missing) for r in results)
    fp = sum(len(r.extra) for r in results)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)

    by_tag: dict[str, dict] = {}
    for r in results:
        for tag in r.case.tags:
            slot = by_tag.setdefault(tag, {"n": 0, "exact": 0, "silent_wrong": 0})
            slot["n"] += 1
            slot["exact"] += int(r.exact)
            slot["silent_wrong"] += int(r.silent_wrong)
    for slot in by_tag.values():
        slot["exact_rate"] = _rate(slot["exact"], slot["n"])

    return {
        "n": n,
        "errors": sum(1 for r in results if r.error),
        "intent_accuracy": _rate(sum(r.intent_ok for r in results), n),
        "exact_match": _rate(sum(r.exact for r in results), n),
        "predicate_precision": precision,
        "predicate_recall": recall,
        "predicate_f1": _rate(2 * precision * recall, precision + recall)
                        if (precision + recall) else 0.0,
        # Headline safety metrics -- these are the ones to hold the line on.
        "silent_wrong": sum(r.silent_wrong for r in results),
        "silent_wrong_rate": _rate(sum(r.silent_wrong for r in results), n),
        "fabricated_filters": fp,
        "over_clarify": sum(r.over_clarify for r in results),
        "by_tag": dict(sorted(by_tag.items())),
    }


def format_report(summary: dict, results: list[CaseResult] | None = None,
                  show_failures: int = 20) -> str:
    lines = [
        "=" * 66,
        f"  cases                {summary['n']}",
        f"  intent accuracy      {summary['intent_accuracy']:.1%}",
        f"  exact match          {summary['exact_match']:.1%}",
        f"  predicate P / R / F1 {summary['predicate_precision']:.1%} / "
        f"{summary['predicate_recall']:.1%} / {summary['predicate_f1']:.1%}",
        "-" * 66,
        f"  SILENT WRONG         {summary['silent_wrong']} "
        f"({summary['silent_wrong_rate']:.1%})   <- guessed instead of asking",
        f"  fabricated filters   {summary['fabricated_filters']}"
        "        <- filters tracing to nothing in the query",
        f"  over-clarified       {summary['over_clarify']}"
        "        <- asked when the query was explicit",
    ]
    if summary["errors"]:
        lines.append(f"  harness/LLM errors   {summary['errors']}")
    lines.append("-" * 66)
    lines.append("  by tag (exact-match rate):")
    for tag, slot in summary["by_tag"].items():
        flag = "  !! silent_wrong=%d" % slot["silent_wrong"] if slot["silent_wrong"] else ""
        lines.append(f"    {tag:<28} {slot['exact']:>3}/{slot['n']:<3} "
                     f"{slot['exact_rate']:>6.1%}{flag}")
    lines.append("=" * 66)

    if results:
        failures = [r for r in results if not r.exact]
        if failures:
            lines.append(f"\nFAILURES ({len(failures)} total, showing "
                         f"{min(show_failures, len(failures))}):")
            for r in failures[:show_failures]:
                lines.append(f"\n  [{r.case.id}] {r.case.query!r}")
                if r.error:
                    lines.append(f"    ERROR: {r.error}")
                    continue
                if not r.intent_ok:
                    lines.append(f"    intent: expected {r.case.expected_intent}, "
                                 f"got {r.predicted.get('intent')}")
                for m in r.missing:
                    lines.append(f"    MISSING  {json.dumps(m)}")
                for e in r.extra:
                    lines.append(f"    EXTRA    {json.dumps(e)}")
                for p in r.metadata_problems:
                    lines.append(f"    META     {p}")
                if r.silent_wrong:
                    lines.append("    !! SILENT WRONG -- guessed instead of asking")
    return "\n".join(lines)
