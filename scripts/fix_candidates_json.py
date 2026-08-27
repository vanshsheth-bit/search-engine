"""One-off: convert data/candidates.json from Mongo-shell pseudo-JSON into
valid JSON. Source uses unquoted keys, single-quoted strings, ObjectId(...)
and ISODate(...) wrappers, and concatenated top-level objects with no
enclosing array -- none of that is valid JSON or even valid JSON5 as-is.

Run once: .venv/Scripts/python.exe scripts/fix_candidates_json.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import json5

SRC = Path(__file__).resolve().parent.parent / "data" / "candidates.json"


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    # Strip Mongo shell type wrappers down to their inner literal.
    text = re.sub(r"ObjectId\('([^']*)'\)", r'"\1"', text)
    text = re.sub(r"ISODate\('([^']*)'\)", r'"\1"', text)
    text = re.sub(r"NumberLong\('?(-?\d+)'?\)", r"\1", text)
    text = re.sub(r"NumberInt\('?(-?\d+)'?\)", r"\1", text)
    text = re.sub(r"Double\('?(-?[\d.]+)'?\)", r"\1", text)

    # The file is N top-level objects separated by "},\n{" with no
    # enclosing array -- wrap it into one.
    wrapped = f"[{text}]"

    records = json5.loads(wrapped)
    print(f"parsed {len(records)} records")

    out_path = SRC.parent / "candidates_raw.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
