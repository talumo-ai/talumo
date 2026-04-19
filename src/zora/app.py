"""FastAPI application — single endpoint that hides backend complexity."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from zora.orchestrator import orchestrate
from zora.schemas import OrchRequest, OrchResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = httpx.AsyncClient()
    yield
    await app.state.client.aclose()


app = FastAPI(title="Zora Orchestrator", version="0.1.0", lifespan=lifespan)


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
