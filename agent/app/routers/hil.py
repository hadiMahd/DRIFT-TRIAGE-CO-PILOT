"""Human-in-the-loop approval HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from agent.app.schemas.hil_action import ApprovalDecisionResponse, HILAction
from agent.app.services import request_approval


router = APIRouter(prefix="/hil", tags=["hil"])


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
