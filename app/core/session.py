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


class InMemorySessionStore(SessionStore):
    def __init__(self, ttl: Optional[int] = None) -> None:
        self._ttl = ttl or settings.session_ttl
        self._data: dict[str, tuple[float, SessionState]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(session_id: str, job_id: str) -> str:
        return f"{session_id}::{job_id}"

    def _expired(self, ts: float) -> bool:
        return (time.time() - ts) > self._ttl

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
        with self._lock:
            self._data[key] = (time.time(), state.model_copy(deep=True))

    def clear(self, session_id: str, job_id: str) -> None:
        with self._lock:
            self._data.pop(self._key(session_id, job_id), None)


# A module-level default store used by the service layer.
default_store = InMemorySessionStore()
