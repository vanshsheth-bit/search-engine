"""Manually set/correct one company's type in the persistent classification
cache -- a human with real, specific knowledge of a company (especially a
small/obscure one the model honestly can't know) should always be able to
override its classification. Persists immediately, same file the live app
reads, takes effect on the next request (no restart needed).

Run: .venv/Scripts/python.exe scripts/set_company_type.py "Rebee.AI" Product
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from app.core.company_type import CATEGORIES, _cache  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: set_company_type.py <company name> <{'/'.join(CATEGORIES)}>")
        sys.exit(1)

    name, category = sys.argv[1], sys.argv[2]
    if category not in CATEGORIES:
        print(f"Category must be one of: {', '.join(CATEGORIES)} (got {category!r})")
        sys.exit(1)

    _cache().update({name.strip().lower(): category})
    print(f"Set {name!r} -> {category}")


if __name__ == "__main__":
    main()
