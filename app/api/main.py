"""FastAPI application exposing the candidate filtering endpoints."""
from __future__ import annotations

import logging
import os

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


@app.post("/ai/candidates/filter", response_model=FilterResponse)
def filter_candidates(req: FilterRequest) -> FilterResponse:
    """Natural-language filter. Returns ok / clarify / unsupported / no_match."""
    return service.filter_by_query(
        query=req.query,
        job_id=req.job_id,
        session_id=req.session_id,
        reset=req.reset,
    )


@app.patch("/ai/candidates/filter/state", response_model=FilterResponse)
def patch_state(req: PatchStateRequest) -> FilterResponse:
    """Deterministic filter edit (e.g. chip removal). No LLM involved."""
    return service.patch_state(req)


@app.delete("/ai/candidates/filter/state", response_model=FilterResponse)
def clear_state(job_id: str, session_id: str) -> FilterResponse:
    """Clear all filters for a session/job and return the full list."""
    return service.clear(session_id=session_id, job_id=job_id)
