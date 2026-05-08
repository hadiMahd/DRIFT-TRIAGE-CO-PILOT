"""Persistence helpers for Human-in-the-Loop approval rows."""

from __future__ import annotations

import importlib
from typing import Any, Literal, Mapping
from uuid import uuid4

from agent.app.schemas.hil_action import HILAction
from agent.app.schemas.investigation import RecommendedAction


ApprovalTransition = Literal["approved", "rejected"]


def _get_postgres_dsn() -> str:
    """Read the Postgres DSN lazily to avoid import-time dependency failures."""

    from agent.app.config.settings import get_settings

    return get_settings().POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://", 1)


def build_idempotency_key(
    requested_action: RecommendedAction,
    investigation_id: str,
    drift_event_id: str,
    target_model_version: str | None = None,
) -> str:
    """Build the stable idempotency key for approval requests."""

    return f"{requested_action}:{investigation_id}:{target_model_version or drift_event_id}"


def _load_asyncpg() -> Any:
    """Import asyncpg only when the service is called."""

    try:
        return importlib.import_module("asyncpg")
    except ImportError as exc:
        raise RuntimeError(
            "asyncpg is required for HIL approval persistence. "
            "Install agent dependencies before calling request_approval service functions."
        ) from exc


async def _connect() -> Any:
    """Open a new asyncpg connection."""

    asyncpg = _load_asyncpg()
    return await asyncpg.connect(_get_postgres_dsn())


def row_to_hil_action(row: Mapping[str, Any] | None) -> HILAction | None:
    """Convert a database row to the public HILAction model."""

    if row is None:
        return None
    row_dict = dict(row)
    payload = {name: row_dict.get(name) for name in HILAction.model_fields}
    return HILAction.model_validate(payload)


def validate_status_transition(current_status: str, target_status: ApprovalTransition) -> None:
    """Ensure only pending approvals can change state."""

    if current_status == target_status:
        return
    if current_status != "pending":
        raise ValueError(
            f"Cannot transition approval from '{current_status}' to '{target_status}'."
        )


async def _fetch_approval_row(connection: Any, approval_id: str) -> Mapping[str, Any] | None:
    """Fetch one approval row by primary key."""

    return await connection.fetchrow(
        """
        SELECT approval_id, investigation_id, drift_event_id, requested_action,
               target_model_version, status, requested_by, approved_by,
               idempotency_key, created_at, updated_at
        FROM hil_approvals
        WHERE approval_id = $1
        """,
        approval_id,
    )


async def create_pending_approval(
    investigation_id: str,
    drift_event_id: str,
    requested_action: RecommendedAction,
    target_model_version: str | None = None,
    requested_by: str = "agent",
    idempotency_key: str | None = None,
) -> HILAction:
    """Insert a pending approval row or return the existing idempotent row."""

    approval_id = str(uuid4())
    stable_key = idempotency_key or build_idempotency_key(
        requested_action=requested_action,
        investigation_id=investigation_id,
        drift_event_id=drift_event_id,
        target_model_version=target_model_version,
    )
    connection = await _connect()
    try:
        row = await connection.fetchrow(
            """
            INSERT INTO hil_approvals (
                approval_id, investigation_id, drift_event_id, requested_action,
                target_model_version, requested_by, idempotency_key
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (idempotency_key) DO UPDATE
            SET idempotency_key = hil_approvals.idempotency_key
            RETURNING approval_id, investigation_id, drift_event_id, requested_action,
                      target_model_version, status, requested_by, approved_by,
                      idempotency_key, created_at, updated_at
            """,
            approval_id,
            investigation_id,
            drift_event_id,
            requested_action,
            target_model_version,
            requested_by,
            stable_key,
        )
        approval = row_to_hil_action(row)
        if approval is None:
            raise RuntimeError("Failed to create or fetch pending approval.")
        return approval
    finally:
        await connection.close()


async def get_approval(approval_id: str) -> HILAction | None:
    """Fetch an approval by id."""

    connection = await _connect()
    try:
        row = await _fetch_approval_row(connection, approval_id)
        return row_to_hil_action(row)
    finally:
        await connection.close()


async def _transition_approval(
    approval_id: str,
    target_status: ApprovalTransition,
    approved_by: str,
    reason: str | None = None,
) -> HILAction:
    """Apply an approval status transition with validation."""

    connection = await _connect()
    try:
        current_row = await _fetch_approval_row(connection, approval_id)
        if current_row is None:
            raise ValueError(f"Approval '{approval_id}' was not found.")

        current = row_to_hil_action(current_row)
        if current is None:
            raise RuntimeError(f"Approval '{approval_id}' could not be parsed.")

        validate_status_transition(current.status, target_status)
        if current.status == target_status:
            return current

        updated_row = await connection.fetchrow(
            """
            UPDATE hil_approvals
            SET status = $2,
                approved_by = $3,
                reason = $4,
                updated_at = NOW()
            WHERE approval_id = $1
            RETURNING approval_id, investigation_id, drift_event_id, requested_action,
                      target_model_version, status, requested_by, approved_by,
                      idempotency_key, created_at, updated_at
            """,
            approval_id,
            target_status,
            approved_by,
            reason,
        )
        approval = row_to_hil_action(updated_row)
        if approval is None:
            raise RuntimeError(f"Approval '{approval_id}' could not be updated.")
        return approval
    finally:
        await connection.close()


async def approve_action(
    approval_id: str,
    approved_by: str,
    reason: str | None = None,
) -> HILAction:
    """Approve a pending HIL action or return the already-approved row."""

    return await _transition_approval(
        approval_id=approval_id,
        target_status="approved",
        approved_by=approved_by,
        reason=reason,
    )


async def reject_action(
    approval_id: str,
    approved_by: str,
    reason: str | None = None,
) -> HILAction:
    """Reject a pending HIL action or return the already-rejected row."""

    return await _transition_approval(
        approval_id=approval_id,
        target_status="rejected",
        approved_by=approved_by,
        reason=reason,
    )


async def list_pending_approvals(limit: int = 50) -> list[HILAction]:
    """List the newest pending approvals."""

    connection = await _connect()
    try:
        rows = await connection.fetch(
            """
            SELECT approval_id, investigation_id, drift_event_id, requested_action,
                   target_model_version, status, requested_by, approved_by,
                   idempotency_key, created_at, updated_at
            FROM hil_approvals
            WHERE status = 'pending'
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [approval for row in rows if (approval := row_to_hil_action(row)) is not None]
    finally:
        await connection.close()
