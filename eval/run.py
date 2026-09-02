"""Run the eval corpus and print a report.

Two modes:

  --self-check   Scores the corpus against ITS OWN expected answers (as if
                 a perfect predictor produced exactly what's expected). This
                 needs no Ollama and no model -- it exists to catch mistakes
                 IN the corpus (a value_includes typo, a missing
                 clarify_field, an id collision) before ever spending a
                 model call on it. Should print 100% on every metric; any
                 other number means a corpus bug, not a model failure.
                 Run this after any corpus edit.

  --live         Calls the real LLMClient (Ollama must be running, MODEL
                 configured via .env same as the server) for every case and
                 scores its actual output. This is the number that answers
                 "how good is qwen3:8b at this task right now" -- something
                 SESSION_NOTES.md never actually measured; every finding
                 there came from testing single queries by hand.

Usage:
  .venv/Scripts/python.exe eval/run.py --self-check
  .venv/Scripts/python.exe eval/run.py --live
  .venv/Scripts/python.exe eval/run.py --live --tag skill_concept
  .venv/Scripts/python.exe eval/run.py --live --failures 50
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from eval.harness import (  # noqa: E402
    Case, CaseResult, format_report, load_corpus, score_case, summarise,
)


def _llm_output_to_dict(out) -> dict:
    """LLMOutput (a pydantic model) -> the plain dict shape the harness
    scores against. Kept as a separate, tiny mapping function so a future
    interface (tool calls, a query tree) only needs a new version of THIS
    function, not a rewritten harness or corpus."""
    return {
        "intent": out.intent,
        "logic": out.logic,
        "filters": [f.model_dump(exclude_none=True) for f in out.filters],
        "clarify_field": out.clarify_field,
        "clarify_skill": out.clarify_skill,
        "clarify_operator": out.clarify_operator,
        "clarify_value": out.clarify_value,
        "clarify_unit": out.clarify_unit,
        "message": out.message,
        "candidate_ref": out.candidate_ref,
        "lookup_field": out.lookup_field,
    }


def run_self_check(cases: list[Case]) -> list[CaseResult]:
    """Predicts exactly the expected answer for every case (plus, for
    UNSUPPORTED_FILTER cases that assert message_mentions, a synthesized
    message containing every required substring, since the corpus doesn't
    store literal message text -- only what it must mention)."""
    results = []
    for c in cases:
        predicted: dict = {"intent": c.expected_intent}
        if c.expected_intent == "FILTER_CANDIDATES":
            predicted["logic"] = c.expect.get("logic", "AND")
            preds = []
            for p in c.expect.get("predicates", []):
                pred = {"field": p["field"], "operator": p["operator"]}
                if "value" in p:
                    pred["value"] = p["value"]
                elif "value_includes" in p:
                    pred["value"] = list(p["value_includes"])
                if "skill" in p:
                    pred["skill"] = p["skill"]
                if "unit" in p:
                    pred["unit"] = p["unit"]
                preds.append(pred)
            predicted["filters"] = preds
        for key in ("clarify_field", "clarify_operator", "clarify_value",
                    "clarify_skill", "clarify_unit", "lookup_field"):
            if key in c.expect:
                predicted[key] = c.expect[key]
        if c.expect.get("message_mentions"):
            predicted["message"] = " -- ".join(c.expect["message_mentions"])
        results.append(score_case(c, predicted))
    return results


def run_live(cases: list[Case]) -> list[CaseResult]:
    from app.llm.client import LLMClient  # noqa: E402

    client = LLMClient()
    results = []
    for i, c in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {c.id}: {c.query!r}", file=sys.stderr)
        try:
            out = client.translate(c.query, c.current_filters, c.history or None)
            predicted = _llm_output_to_dict(out)
            results.append(score_case(c, predicted))
        except Exception as exc:  # noqa: BLE001 -- report, don't abort the run
            results.append(score_case(c, {}, error=str(exc)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-check", action="store_true",
                       help="Score the corpus against itself (no model needed).")
    mode.add_argument("--live", action="store_true",
                       help="Score the real LLMClient (needs Ollama running).")
    parser.add_argument("--tag", action="append", default=[],
                         help="Only run cases carrying this tag (repeatable).")
    parser.add_argument("--failures", type=int, default=20,
                         help="Max failure entries to print (default 20).")
    args = parser.parse_args()

    cases = load_corpus()
    if args.tag:
        wanted = set(args.tag)
        cases = [c for c in cases if wanted & set(c.tags)]
        if not cases:
            print(f"No cases match tag(s) {sorted(wanted)}", file=sys.stderr)
            sys.exit(1)

    results = run_self_check(cases) if args.self_check else run_live(cases)
    summary = summarise(results)
    print(format_report(summary, results, show_failures=args.failures))

    if args.self_check and summary["exact_match"] < 1.0:
        print("\nSELF-CHECK FAILED -- the corpus disagrees with its own "
              "expectations. Fix the corpus before trusting any --live run.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
