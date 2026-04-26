"""FastAPI application — single endpoint that hides backend complexity."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from talumo.orchestrator import orchestrate
from talumo.schemas import OrchRequest, OrchResponse
from talumo.settings import settings


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. Local tier can run, but remote tiers will fail if used."
        )
    app.state.client = httpx.AsyncClient()
    yield
    await app.state.client.aclose()


app = FastAPI(title="Talumo Orchestrator", version="0.1.0", lifespan=lifespan)


@app.post("/v1/orchestrate", response_model=OrchResponse)
async def handle_orchestrate(body: OrchRequest, request: Request) -> OrchResponse:
    client: httpx.AsyncClient = request.app.state.client
    return await orchestrate(body, client)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal orchestrator error", "type": type(exc).__name__},
    )
