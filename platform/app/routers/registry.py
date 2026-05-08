"""Registry endpoints — promotion gate, status, audit, rollback, history.

POST /promote   — promotion gate with approval_id validation
POST /rollback  — immediate rollback to a previous version
GET /status     — current Production/Candidate versions with metrics
GET /history    — promotion audit trail
"""

import json
from datetime import datetime, timezone

import httpx
import mlflow
from fastapi import APIRouter, Depends, HTTPException, Request
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict

from app.dependencies import get_http_client, get_settings
from app.schemas.promote_request import PromoteRequest
from app.services.validate_promotion import assert_promotion_checklist

router = APIRouter()


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_version: str
    approval_id: str
    approved_by: str


def _pg_dsn(settings) -> str:
    return settings.postgres_dsn.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _fetch_production_metrics(settings) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    try:
        prod = client.get_model_version_by_alias(
            settings.registered_model_name, "Production",
        )
        run_id = prod.run_id
        if run_id:
            run = client.get_run(run_id)
            m = run.data.metrics
            return {
                "test_recall": m.get("test_recall", 0),
                "test_f1": m.get("test_f1", 0),
                "test_roc_auc": m.get("test_roc_auc", 0),
                "operating_threshold": m.get("operating_threshold", 0),
            }
    except Exception:
        pass
    return {}


async def _previous_production_version(settings) -> str | None:
    try:
        import asyncpg
        conn = await asyncpg.connect(_pg_dsn(settings), timeout=5)
        try:
            row = await conn.fetchrow(
                """SELECT previous_version FROM promotion_audit
                   WHERE previous_version IS NOT NULL
                   ORDER BY timestamp DESC LIMIT 1"""
            )
            if row:
                return row["previous_version"]
        finally:
            await conn.close()
    except Exception:
        pass
    return None


def _capture_current_production(settings) -> str | None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    try:
        prod = client.get_model_version_by_alias(
            settings.registered_model_name, "Production",
        )
        return str(prod.version)
    except Exception:
        return None


async def _reload_model(request: Request, settings) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        model = mlflow.sklearn.load_model(
            f"models:/{settings.registered_model_name}@Production"
        )
        request.app.state.model = model

        client = MlflowClient()
        prod = client.get_model_version_by_alias(
            settings.registered_model_name, "Production",
        )
        if prod.run_id:
            run = client.get_run(prod.run_id)
            request.app.state.threshold = run.data.metrics.get(
                "operating_threshold", settings.threshold,
            )
    except Exception:
        pass


@router.get("/status")
async def registry_status(settings=Depends(get_settings)) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    prod_ver: str | None = None
    cand_ver: str | None = None
    last_promotion: str | None = None

    try:
        prod = client.get_model_version_by_alias(
            settings.registered_model_name, "Production",
        )
        prod_ver = str(prod.version)
    except Exception:
        pass

    try:
        cand = client.get_model_version_by_alias(
            settings.registered_model_name, "candidate",
        )
        cand_ver = str(cand.version)
        if cand.last_updated_timestamp:
            last_promotion = datetime.fromtimestamp(
                cand.last_updated_timestamp / 1000, tz=timezone.utc,
            ).isoformat()
    except Exception:
        pass

    production_metrics = await _fetch_production_metrics(settings) if prod_ver else {}
    previous_version = await _previous_production_version(settings)

    return {
        "registered_model_name": settings.registered_model_name,
        "production_version": prod_ver,
        "production_metrics": production_metrics,
        "previous_production_version": previous_version,
        "candidate_version": cand_ver,
        "last_promotion": last_promotion,
        "status": "ok",
    }


@router.post("/promote")
async def promote(
    body: PromoteRequest,
    client: httpx.AsyncClient = Depends(get_http_client),
    settings=Depends(get_settings),
    request: Request = None,
) -> dict:
    if not body.approved_by:
        raise HTTPException(
            status_code=422,
            detail="approved_by is required — promotion must be authorized.",
        )

    try:
        assert_promotion_checklist(body.model_uri)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = MlflowClient()

    previous_version = _capture_current_production(settings)

    try:
        await _write_audit_to_postgres(settings, body, previous_version)

        mlflow_client.set_registered_model_alias(
            name=settings.registered_model_name,
            alias="Production",
            version=body.model_uri.split("/")[-1],
        )

        if request:
            await _reload_model(request, settings)

        audit_row = {
            "investigation_id": body.investigation_id,
            "approved_by": body.approved_by,
            "model_uri": body.model_uri,
            "previous_version": previous_version,
            "timestamp": body.timestamp.isoformat(),
        }
        logger = __import__("structlog").get_logger()
        logger.info("promotion_audit", **audit_row)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Promotion failed at registry level: {exc}",
        ) from exc

    return {
        "status": "promoted",
        "model_uri": body.model_uri,
        "approved_by": body.approved_by,
    }


@router.post("/rollback")
async def rollback(
    body: RollbackRequest,
    settings=Depends(get_settings),
    request: Request = None,
) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow_client = MlflowClient()

    approval = await _validate_rollback_approval(settings, body)

    try:
        mlflow_client.get_model_version(
            name=settings.registered_model_name,
            version=body.target_version,
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Model version {body.target_version} not found in registry.",
        )

    previous_version = _capture_current_production(settings)

    try:
        await _write_rollback_audit(settings, body, approval, previous_version)

        mlflow_client.set_registered_model_alias(
            name=settings.registered_model_name,
            alias="Production",
            version=body.target_version,
        )

        if request:
            await _reload_model(request, settings)

        logger = __import__("structlog").get_logger()
        logger.info(
            "rollback_executed",
            target_version=body.target_version,
            previous_version=previous_version,
            approval_id=body.approval_id,
            approved_by=body.approved_by,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Rollback failed at registry level: {exc}",
        ) from exc

    return await registry_status(settings)


@router.get("/history")
async def registry_history(settings=Depends(get_settings)) -> dict:
    records: list[dict] = []
    try:
        import asyncpg
        conn = await asyncpg.connect(_pg_dsn(settings), timeout=5)
        try:
            rows = await conn.fetch(
                """SELECT model_uri, investigation_id, approved_by, timestamp,
                          from_alias, to_alias, previous_version
                   FROM promotion_audit
                   ORDER BY timestamp DESC
                   LIMIT 50"""
            )
            for row in rows:
                records.append({
                    "model_uri": row["model_uri"],
                    "investigation_id": row["investigation_id"],
                    "approved_by": row["approved_by"],
                    "timestamp": row["timestamp"].isoformat(),
                    "from_alias": row["from_alias"],
                    "to_alias": row["to_alias"],
                    "previous_version": row["previous_version"],
                })
        finally:
            await conn.close()
    except Exception:
        pass

    return {"history": records}


async def _write_audit_to_postgres(settings, body, previous_version: str | None = None) -> None:
    import asyncpg

    conn = await asyncpg.connect(_pg_dsn(settings), timeout=5)
    try:
        await conn.execute(
            "INSERT INTO promotion_audit (model_uri, investigation_id, approved_by, from_alias, to_alias, previous_version) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            body.model_uri, body.investigation_id, body.approved_by,
            "candidate", "Production", previous_version,
        )
    finally:
        await conn.close()


async def _write_rollback_audit(settings, body: RollbackRequest, approval: dict, previous_version: str | None = None) -> None:
    import asyncpg

    conn = await asyncpg.connect(_pg_dsn(settings), timeout=5)
    try:
        await conn.execute(
            "INSERT INTO promotion_audit (model_uri, investigation_id, approved_by, from_alias, to_alias, previous_version) "
            "VALUES ($1, $2, $3, $4, $5, $6)",
            body.target_version, approval["investigation_id"], body.approved_by,
            "Production", "Production", previous_version,
        )
    finally:
        await conn.close()


async def _validate_rollback_approval(settings, body: RollbackRequest) -> dict:
    if not body.approval_id:
        raise HTTPException(
            status_code=422,
            detail="approval_id is required for rollback.",
        )

    try:
        import asyncpg

        conn = await asyncpg.connect(_pg_dsn(settings), timeout=5)
        try:
            row = await conn.fetchrow(
                """SELECT approval_id, investigation_id, requested_action,
                          target_model_version, status, approved_by
                   FROM hil_approvals
                   WHERE approval_id = $1""",
                body.approval_id,
            )
        finally:
            await conn.close()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Rollback approval check failed: {exc}",
        ) from exc

    if row is None:
        raise HTTPException(status_code=404, detail="Rollback approval not found.")
    if row["status"] != "approved":
        raise HTTPException(status_code=409, detail="Rollback approval is not approved.")
    if row["requested_action"] != "rollback":
        raise HTTPException(status_code=409, detail="Approval is not for rollback.")
    if row["target_model_version"] and str(row["target_model_version"]) != str(body.target_version):
        raise HTTPException(
            status_code=409,
            detail="Approval target version does not match rollback target.",
        )

    return dict(row)
