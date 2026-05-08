"""FastAPI application assembly.

Lifespan:
- Load model.joblib and operating threshold from disk
- Create shared httpx client
- Store in app.state.model, app.state.threshold, app.state.http_client

Routers mounted:
- /predict  → routers/predict.py
- /drift    → routers/drift.py
- /registry → routers/registry.py
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.config.settings import Settings
from app.dependencies import load_model
from app.routers import drift, predict, queue, registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.model = load_model(settings.resolved_model_path())
    app.state.threshold = settings.threshold
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    app.state.drift_accumulator: list[dict] = []
    app.state.last_severity: str = "stable"
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Drift Triage Platform", version="0.1.0", lifespan=lifespan)

app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(drift.router, prefix="/drift", tags=["drift"])
app.include_router(registry.router, prefix="/registry", tags=["registry"])
app.include_router(queue.router, prefix="/queue", tags=["queue"])


@app.get("/health")
async def health():
    return {"status": "ok"}
