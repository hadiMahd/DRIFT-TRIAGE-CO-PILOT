"""Human-in-the-loop approval HTTP endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from agent.app.config.settings import get_settings
from agent.app.schemas.hil_action import ApprovalDecisionResponse, HILAction
from agent.app.services import investigations, request_approval


router = APIRouter(prefix="/hil", tags=["hil"])
REGISTERED_MODEL_NAME = "bank_marketing_pipeline"


class HILDecisionBody(BaseModel):
    """Minimal decision body sent by a human approver."""

    model_config = ConfigDict(extra="forbid")

    approved_by: str
    reason: str | None = None


def _not_found(approval_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"message": "Approval not found", "approval_id": approval_id},
    )


def _service_unavailable(message: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"message": message},
    )


def _internal_error() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail={"message": "Internal approval error"},
    )


def _platform_error(status_code: int, detail: object) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"message": "Platform action failed", "platform_detail": detail},
    )


@router.get("/pending")
async def get_pending_approvals(limit: int = 50) -> dict[str, list[dict]]:
    """Return pending approvals using the existing persistence service."""

    try:
        approvals = await request_approval.list_pending_approvals(limit=limit)
    except RuntimeError as exc:
        raise _service_unavailable(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for route stability
        raise _internal_error() from exc

    return {
        "approvals": [approval.model_dump(mode="json") for approval in approvals],
    }


@router.get("/{approval_id}")
async def get_approval(approval_id: str) -> dict:
    """Return one approval by id."""

    try:
        approval = await request_approval.get_approval(approval_id)
    except RuntimeError as exc:
        raise _service_unavailable(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for route stability
        raise _internal_error() from exc

    if approval is None:
        raise _not_found(approval_id)

    return approval.model_dump(mode="json")


@router.post("/{approval_id}/approve", response_model=ApprovalDecisionResponse)
async def approve_approval(
    approval_id: str,
    body: HILDecisionBody,
) -> ApprovalDecisionResponse:
    """Approve a pending HIL action."""

    try:
        approval = await request_approval.get_approval(approval_id)
        if approval is None:
            raise _not_found(approval_id)
        updated = await request_approval.approve_action(
            approval_id=approval_id,
            approved_by=body.approved_by,
            reason=body.reason,
        )
        await _apply_approved_action(updated, body.approved_by)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
    except RuntimeError as exc:
        raise _service_unavailable(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for route stability
        raise _internal_error() from exc

    return ApprovalDecisionResponse(
        approval_id=updated.approval_id,
        status=updated.status,
        message="Approval approved",
    )


@router.post("/{approval_id}/reject", response_model=ApprovalDecisionResponse)
async def reject_approval(
    approval_id: str,
    body: HILDecisionBody,
) -> ApprovalDecisionResponse:
    """Reject a pending HIL action."""

    try:
        approval = await request_approval.get_approval(approval_id)
        if approval is None:
            raise _not_found(approval_id)
        updated = await request_approval.reject_action(
            approval_id=approval_id,
            approved_by=body.approved_by,
            reason=body.reason,
        )
        await _mark_rejected(updated)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc
    except RuntimeError as exc:
        raise _service_unavailable(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for route stability
        raise _internal_error() from exc

    return ApprovalDecisionResponse(
        approval_id=updated.approval_id,
        status=updated.status,
        message="Approval rejected",
    )


class NotifyCandidateBody(BaseModel):
    """Payload from the worker after a new candidate model is trained."""
    model_config = ConfigDict(extra="forbid")

    investigation_id: str
    drift_event_id: str
    model_uri: str
    target_model_version: str | None = None
    metrics: dict[str, float] | None = None


@router.post("/notify-candidate", status_code=201)
async def notify_candidate_ready(body: NotifyCandidateBody) -> dict:
    """Called by the worker when retrain completes."""
    version = body.target_model_version or body.model_uri.split("/")[-1]

    try:
        approval = await request_approval.create_pending_approval(
            investigation_id=body.investigation_id,
            drift_event_id=body.drift_event_id,
            requested_action="promote_candidate",
            target_model_version=version,
            requested_by="worker",
            idempotency_key=f"worker:{body.investigation_id}:{version}",
        )
    except RuntimeError as exc:
        raise _service_unavailable(str(exc)) from exc

    return {
        "approval_id": approval.approval_id,
        "status": approval.status,
        "model_uri": body.model_uri,
        "metrics": body.metrics or {},
        "message": "HIL approval created — candidate ready for review",
    }


async def _apply_approved_action(approval: HILAction, approved_by: str) -> None:
    if approval.requested_action == "promote_candidate":
        resolved_version = await _resolve_target_version(approval)
        await _dispatch_promotion(approval, approved_by, resolved_version)
        await _record_resolution(
            approval,
            approved_by=approved_by,
            status="resolved",
            summary=(
                f"Human approval executed promotion of candidate version "
                f"{resolved_version}."
            ),
        )
        return

    if approval.requested_action == "rollback":
        resolved_version = await _resolve_target_version(approval)
        await _dispatch_rollback(approval, approved_by, resolved_version)
        await _record_resolution(
            approval,
            approved_by=approved_by,
            status="resolved",
            summary=(
                f"Human approval executed rollback to version "
                f"{resolved_version}."
            ),
        )
        return

    await _record_resolution(
        approval,
        approved_by=approved_by,
        status="approved",
        summary=f"Human approval recorded for action '{approval.requested_action}'.",
    )


async def _mark_rejected(approval: HILAction) -> None:
    await _record_resolution(
        approval,
        approved_by=approval.approved_by or "human",
        status="rejected",
        summary=f"Human approval rejected action '{approval.requested_action}'.",
    )


async def _record_resolution(
    approval: HILAction,
    *,
    approved_by: str,
    status: str,
    summary: str,
) -> None:
    try:
        state = await investigations.load_state_by_investigation_id(approval.investigation_id)
        if state is None:
            return
        state["status"] = status
        state["recommended_action"] = approval.requested_action
        state["approval_id"] = approval.approval_id
        state["requires_approval"] = False
        state["queued"] = False
        state["dispatch_error"] = None
        state["comms_summary"] = summary
        await investigations.save_state(state, last_completed_node="hil_decision")
    except Exception:
        pass


async def _dispatch_promotion(
    approval: HILAction,
    approved_by: str,
    resolved_version: str,
) -> None:
    settings = get_settings()
    payload = {
        "model_uri": await _build_model_uri(resolved_version),
        "approved_by": approved_by,
        "approval_id": approval.approval_id,
        "investigation_id": approval.investigation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{settings.PLATFORM_BASE_URL}/registry/promote", json=payload)
    if response.status_code >= 400:
        detail = _response_detail(response)
        await _record_failed_dispatch(approval, detail)
        raise _platform_error(response.status_code, detail)


async def _dispatch_rollback(
    approval: HILAction,
    approved_by: str,
    resolved_version: str,
) -> None:
    settings = get_settings()
    payload = {
        "target_version": resolved_version,
        "approval_id": approval.approval_id,
        "approved_by": approved_by,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{settings.PLATFORM_BASE_URL}/registry/rollback", json=payload)
    if response.status_code >= 400:
        detail = _response_detail(response)
        await _record_failed_dispatch(approval, detail)
        raise _platform_error(response.status_code, detail)


async def _record_failed_dispatch(approval: HILAction, detail: object) -> None:
    try:
        state = await investigations.load_state_by_investigation_id(approval.investigation_id)
        if state is None:
            return
        state["status"] = "failed"
        state["approval_id"] = approval.approval_id
        state["requires_approval"] = False
        state["dispatch_error"] = str(detail)
        state["comms_summary"] = (
            f"Human approval was recorded, but platform action '{approval.requested_action}' failed: {detail}."
        )
        await investigations.save_state(state, last_completed_node="hil_dispatch_failed")
    except Exception:
        pass


async def _build_model_uri(version: str) -> str:
    model_name = REGISTERED_MODEL_NAME
    try:
        status = await _fetch_registry_status()
        model_name = status.get("registered_model_name") or model_name
    except HTTPException:
        pass
    return f"models:/{model_name}/{version}"


async def _resolve_target_version(approval: HILAction) -> str:
    raw = (approval.target_model_version or "").strip()
    if raw and raw.isdigit():
        return raw

    status = await _fetch_registry_status()
    if approval.requested_action == "promote_candidate":
        candidate = status.get("candidate_version")
        if candidate:
            return str(candidate)
        raise HTTPException(
            status_code=409,
            detail={"message": "No candidate version is available to promote."},
        )

    if approval.requested_action == "rollback":
        previous = status.get("previous_production_version")
        if previous:
            return str(previous)
        raise HTTPException(
            status_code=409,
            detail={"message": "No previous Production version is available to roll back to."},
        )

    if raw:
        return raw
    raise HTTPException(status_code=409, detail={"message": "Approval target version is missing."})


async def _fetch_registry_status() -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{settings.PLATFORM_BASE_URL}/registry/status")
    if response.status_code >= 400:
        raise _platform_error(response.status_code, _response_detail(response))
    return response.json()


def _response_detail(response: httpx.Response) -> object:
    try:
        body = response.json()
    except ValueError:
        body = response.text.strip() or response.reason_phrase
    return body
