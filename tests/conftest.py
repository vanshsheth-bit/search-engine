"""Shared pytest fixtures.

Autouse, session-wide: redirects app.core.query_log's write target away from
the real query_log.jsonl at the repo root. Real, reported bug this fixes --
every test that drives FilterService.filter_by_query (test_service.py has
dozens) calls the same log_query() a real request would, and with no
redirect it wrote straight into the SAME file real usage would append to.
Confirmed live: the recruiter-facing query_log.jsonl was full of literal
test-only query strings ("mumbai", "python 99 years", "high salary", "yes",
"bangalore", ...) at test-run timestamps, indistinguishable from real
activity and making the log useless for its actual purpose. Patches the
module-level QUERY_LOG_PATH name directly (not an env var re-read) since
log_query() looks it up from module scope at call time, so this takes
effect regardless of when the module was first imported.
"""
from __future__ import annotations

import pytest

import app.core.query_log as query_log_module


@pytest.fixture(autouse=True)
def _redirect_query_log(tmp_path, monkeypatch):
    monkeypatch.setattr(query_log_module, "QUERY_LOG_PATH", tmp_path / "test_query_log.jsonl")
