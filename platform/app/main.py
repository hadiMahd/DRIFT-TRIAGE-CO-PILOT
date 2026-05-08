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
from app.services.drift_state import DriftStateStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.model = load_model(settings.resolved_model_path())
    app.state.threshold = settings.threshold
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        await registry.ensure_registry_schema(settings)
    except Exception:
        pass
    app.state.drift_state_store = None
    try:
        drift_state_store = DriftStateStore(
            postgres_dsn=settings.postgres_dsn,
            window_size=settings.drift_window_size,
        )
        await drift_state_store.ensure_schema()
        runtime_state = await drift_state_store.load_state()
        app.state.drift_state_store = drift_state_store
        app.state.drift_accumulator = runtime_state.accumulator
        app.state.last_severity = runtime_state.last_severity
    except Exception:
        app.state.drift_accumulator = []
        app.state.last_severity = "stable"
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
