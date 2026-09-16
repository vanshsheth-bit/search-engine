"""FastAPI application exposing the candidate filtering endpoints."""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.service import FilterService
from app.models.schemas import (
    FilterRequest,
    FilterResponse,
    PatchStateRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Candidate Filter API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

service = FilterService()

_SEARCH_UI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "search-ui")
if os.path.isdir(_SEARCH_UI_DIR):
    app.mount("/ui", StaticFiles(directory=_SEARCH_UI_DIR, html=True), name="ui")


@app.get("/health")
def health() -> dict:
    """Liveness + Ollama reachability check."""
    ollama_ok = False
    detail = ""
    try:
        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        ollama_ok = r.ok
    except requests.RequestException as exc:
        detail = str(exc)
    return {
        "status": "ok",
        "model": settings.model,
        "ollama_reachable": ollama_ok,
        "detail": detail,
    }


logger = logging.getLogger(__name__)


def _warm_up_ollama() -> None:
    """Fire a trivial completion at server startup so the model is already
    loaded in memory before the first REAL recruiter query arrives.

    Real, reported pain this targets: with no warm-up, the first request
    after a server (re)start pays Ollama's full model-load time on top of
    generation -- confirmed live this session as 240s+ timeouts. Uses the
    SAME num_ctx as the real translate() call (app/llm/client.py) --
    loading with a different context size would just force a SECOND reload
    on the first real request instead of avoiding one (see
    skill_verify.py's own comment on this exact mismatch). num_predict=1
    keeps this to "load the weights and decode one token", not a real
    generation. Best-effort: runs in a background thread so a slow or
    unreachable Ollama never delays server startup, and any failure here
    (Ollama not up yet, etc.) is only logged -- the first real request will
    still retry normally, just without the warm-up's benefit."""
    try:
        requests.post(
            f"{settings.ollama_url}/api/chat",
            json={
                "model": settings.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "think": False,
                "options": {"num_predict": 1, "num_ctx": settings.num_ctx},
                "keep_alive": settings.ollama_keep_alive,
            },
            timeout=settings.llm_timeout,
        )
        logger.info("Ollama warm-up complete (model=%s)", settings.model)
    except requests.RequestException as exc:
        logger.warning("Ollama warm-up failed (will retry on first real request): %s", exc)


@app.on_event("startup")
def _on_startup() -> None:
    threading.Thread(target=_warm_up_ollama, daemon=True).start()


@app.post("/ai/candidates/filter", response_model=FilterResponse)
def filter_candidates(req: FilterRequest) -> FilterResponse:
    """Natural-language filter. Returns ok / clarify / unsupported / no_match."""
    t0_wall = datetime.now().isoformat(timespec="milliseconds")
    t0 = time.perf_counter()
    resp = service.filter_by_query(
        query=req.query,
        job_id=req.job_id,
        session_id=req.session_id,
        reset=req.reset,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "REQUEST /ai/candidates/filter query=%r start=%s end-to-end=%.2fs status=%s",
        req.query, t0_wall, elapsed, resp.status,
    )
    # Plain, simple console line for at-a-glance timing -- the logger.info
    # line above carries the same info but gets buried among the verbose
    # per-request LLM/VALIDATE/APPLY_SPEC log lines above it.
    print(f"[{datetime.now().strftime('%H:%M:%S')}] \"{req.query}\" -> {elapsed:.2f}s")
    return resp


@app.patch("/ai/candidates/filter/state", response_model=FilterResponse)
def patch_state(req: PatchStateRequest) -> FilterResponse:
    """Deterministic filter edit (e.g. chip removal). No LLM involved."""
    return service.patch_state(req)


@app.delete("/ai/candidates/filter/state", response_model=FilterResponse)
def clear_state(job_id: str, session_id: str) -> FilterResponse:
    """Clear all filters for a session/job and return the full list."""
    return service.clear(session_id=session_id, job_id=job_id)
