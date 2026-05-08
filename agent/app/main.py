"""Minimal FastAPI app for the deterministic drift agent."""

from __future__ import annotations

from fastapi import FastAPI

from agent.app.routers import hil, webhook


app = FastAPI(title="Drift Triage Co-Pilot Agent", version="0.1.0")
app.include_router(webhook.router)
app.include_router(hil.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness endpoint."""

    return {
        "status": "ok",
        "service": "agent",
    }
