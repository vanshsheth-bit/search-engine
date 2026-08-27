"""Interactive CLI demo. Requires Ollama running with the model pulled.

    python demo.py

Type natural-language filters; type 'reset' to clear, 'quit' to exit.
"""
from __future__ import annotations

import json

from app.core.service import FilterService

JOB_ID = "123"
SESSION_ID = "cli-session"


def main() -> None:
    svc = FilterService()
    print("Candidate filter demo. Type a filter, 'reset', or 'quit'.\n")
    while True:
        try:
            query = input("filter> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            break

        reset = query.lower() == "reset"
        if reset:
            query = ""  # engine ignores; we just clear
            resp = svc.clear(SESSION_ID, JOB_ID)
        else:
            resp = svc.filter_by_query(query, JOB_ID, SESSION_ID)

        print(f"\nstatus: {resp.status}")
        if resp.chips:
            print("filters: " + "  ".join(c.label for c in resp.chips))
        if resp.status == "clarify":
            print(f"  ? {resp.question}")
            if resp.options:
                print("  options: " + " | ".join(resp.options))
        elif resp.status == "unsupported":
            print(f"  {resp.message}")
        elif resp.status == "no_match":
            print(f"  {resp.message}")
            for s in resp.suggestions:
                print(f"   - {s}")
        elif resp.status == "ok":
            print(f"  showing {resp.showing} of {resp.total}")
            for c in resp.candidates:
                print(f"   [{c['match_score']}] {c['name']} — "
                      f"{c.get('location','?')}")
        print()


if __name__ == "__main__":
    main()
