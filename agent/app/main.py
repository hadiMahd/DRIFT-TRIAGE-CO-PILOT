"""Minimal FastAPI app for the deterministic drift agent."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.app.routers import hil, webhook
from agent.app.services import investigations, request_approval


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await investigations.ensure_tables()
    except Exception:
        pass
    try:
        yield
    finally:
        for close_pool in (request_approval.close_pool, investigations.close_pool):
            try:
                await close_pool()
            except Exception:
                pass


app = FastAPI(title="Drift Triage Co-Pilot Agent", version="0.1.0", lifespan=lifespan)
app.include_router(webhook.router)
app.include_router(hil.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness endpoint."""

    return {
        "status": "ok",
        "service": "agent",
    }
