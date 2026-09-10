"""Session state store.

Holds the current filters AND the last-shown candidate list per
(session_id, job_id) -- the latter is what lets a follow-up like "which
college did he go to" resolve to a real candidate record instead of the LLM
having to invent an answer. Ships with a thread-safe in-memory implementation
with TTL. Swap in Redis for production multi-instance deployments by
implementing the same interface.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from app.core.config import settings
from app.models.schemas import SessionState


class SessionStore:
    def get(self, session_id: str, job_id: str) -> SessionState: ...
    def set(self, session_id: str, job_id: str, state: SessionState) -> None: ...
    def clear(self, session_id: str, job_id: str) -> None: ...


# How often (seconds) `set` sweeps the whole store for expired entries.
# Without a sweep, expiry is purely lazy -- an entry is only ever dropped
# when that SAME key is looked up again -- so a session that is created and
# then abandoned (the common case: one search, browser closed) is retained
# for the life of the process. That is not just a stale key: each entry
# holds a full SessionState including `last_candidates`, i.e. every
# candidate dict that was on screen, so abandoned sessions accumulate real
# memory indefinitely on a long-running server.
_SWEEP_INTERVAL_SECONDS = 60


class InMemorySessionStore(SessionStore):
    def __init__(self, ttl: Optional[int] = None) -> None:
        self._ttl = ttl or settings.session_ttl
        self._data: dict[str, tuple[float, SessionState]] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.time()

    @staticmethod
    def _key(session_id: str, job_id: str) -> str:
        return f"{session_id}::{job_id}"

    def _expired(self, ts: float) -> bool:
        return (time.time() - ts) > self._ttl

    def _sweep_if_due(self, now: float) -> None:
        """Drop every expired entry, at most once per _SWEEP_INTERVAL_SECONDS.
        Caller must hold the lock. Throttled rather than run on every write
        because it is O(number of live sessions) and buys nothing when run
        more often than entries can plausibly expire."""
        if now - self._last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        for key in [k for k, (ts, _) in self._data.items() if self._expired(ts)]:
            del self._data[key]

    def get(self, session_id: str, job_id: str) -> SessionState:
        key = self._key(session_id, job_id)
        with self._lock:
            entry = self._data.get(key)
            if entry is None or self._expired(entry[0]):
                self._data.pop(key, None)
                return SessionState()
            return entry[1].model_copy(deep=True)

    def set(self, session_id: str, job_id: str, state: SessionState) -> None:
        key = self._key(session_id, job_id)
        now = time.time()
        with self._lock:
            self._sweep_if_due(now)
            self._data[key] = (now, state.model_copy(deep=True))

    def clear(self, session_id: str, job_id: str) -> None:
        with self._lock:
            self._data.pop(self._key(session_id, job_id), None)


# A module-level default store used by the service layer.
default_store = InMemorySessionStore()
