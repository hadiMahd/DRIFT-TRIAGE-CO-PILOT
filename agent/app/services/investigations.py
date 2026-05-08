"""Persistence helpers for durable agent investigation state."""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any, Mapping

from agent.app.schemas.drift_alert import DriftAlert
from agent.app.schemas.investigation import Investigation


CHECKPOINT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS investigation_checkpoints (
    investigation_id TEXT PRIMARY KEY,
    drift_event_id TEXT NOT NULL,
    last_completed_node TEXT NULL,
    state_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CHECKPOINT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_investigation_checkpoints_drift_event_id
    ON investigation_checkpoints (drift_event_id);
"""
_pool: Any | None = None
_pool_lock: asyncio.Lock | None = None


def _get_postgres_dsn() -> str:
    from agent.app.config.settings import get_settings

    return get_settings().POSTGRES_DSN.replace("postgresql+asyncpg://", "postgresql://", 1)


def _load_asyncpg() -> Any:
    try:
        return importlib.import_module("asyncpg")
    except ImportError as exc:
        raise RuntimeError(
            "asyncpg is required for investigation persistence. "
            "Install agent dependencies before calling investigation services."
        ) from exc


async def _connect() -> Any:
    asyncpg = _load_asyncpg()
    return await asyncpg.connect(_get_postgres_dsn())


def _get_pool_lock() -> asyncio.Lock:
    global _pool_lock
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    return _pool_lock


async def _get_pool() -> Any:
    global _pool

    if _pool is not None:
        return _pool

    async with _get_pool_lock():
        if _pool is None:
            asyncpg = _load_asyncpg()
            _pool = await asyncpg.create_pool(
                dsn=_get_postgres_dsn(),
                min_size=1,
                max_size=10,
                command_timeout=30,
            )
    return _pool


async def close_pool() -> None:
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None


async def ensure_tables() -> None:
    pool = await _get_pool()
    async with pool.acquire() as connection:
        await connection.execute(CHECKPOINT_TABLE_SQL)
        await connection.execute(CHECKPOINT_INDEX_SQL)


def _serialize_state(state: Mapping[str, Any]) -> str:
    payload = dict(state)
    drift_alert = payload.get("drift_alert")
    if isinstance(drift_alert, DriftAlert):
        payload["drift_alert"] = drift_alert.model_dump(mode="json")
    return json.dumps(payload)


def _deserialize_state(raw_state: Any) -> dict[str, Any] | None:
    if raw_state is None:
        return None
    payload = json.loads(raw_state) if isinstance(raw_state, str) else dict(raw_state)
    drift_alert = payload.get("drift_alert")
    if isinstance(drift_alert, dict):
        payload["drift_alert"] = DriftAlert.model_validate(drift_alert)
    return payload


def _row_to_investigation(row: Mapping[str, Any] | None) -> Investigation | None:
    if row is None:
        return None
    return Investigation.model_validate(dict(row))


async def load_state_by_drift_event(drift_event_id: str) -> dict[str, Any] | None:
    pool = await _get_pool()
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT checkpoint.state_json
            FROM investigation_checkpoints AS checkpoint
            JOIN investigations AS investigations
              ON investigations.investigation_id = checkpoint.investigation_id
            WHERE investigations.drift_event_id = $1
            ORDER BY checkpoint.updated_at DESC
            LIMIT 1
            """,
            drift_event_id,
        )
        if row is None:
            return None
        return _deserialize_state(row["state_json"])


async def load_state_by_investigation_id(investigation_id: str) -> dict[str, Any] | None:
    pool = await _get_pool()
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT state_json
            FROM investigation_checkpoints
            WHERE investigation_id = $1
            """,
            investigation_id,
        )
        if row is None:
            return None
        return _deserialize_state(row["state_json"])


async def get_investigation(investigation_id: str) -> Investigation | None:
    pool = await _get_pool()
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT investigation_id, drift_event_id, status, severity,
                   recommended_action, summary, created_at, updated_at
            FROM investigations
            WHERE investigation_id = $1
            """,
            investigation_id,
        )
        return _row_to_investigation(row)


async def save_state(state: Mapping[str, Any], last_completed_node: str | None = None) -> None:
    summary = state.get("comms_summary") or state.get("triage_summary")
    pool = await _get_pool()
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO investigations (
                investigation_id, drift_event_id, status, severity,
                recommended_action, summary
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (investigation_id) DO UPDATE
            SET status = EXCLUDED.status,
                severity = EXCLUDED.severity,
                recommended_action = EXCLUDED.recommended_action,
                summary = EXCLUDED.summary,
                updated_at = NOW()
            """,
            state["investigation_id"],
            state["drift_event_id"],
            state["status"],
            state["severity"],
            state.get("recommended_action"),
            summary,
        )
        await connection.execute(
            """
            INSERT INTO investigation_checkpoints (
                investigation_id, drift_event_id, last_completed_node, state_json
            )
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (investigation_id) DO UPDATE
            SET drift_event_id = EXCLUDED.drift_event_id,
                last_completed_node = EXCLUDED.last_completed_node,
                state_json = EXCLUDED.state_json,
                updated_at = NOW()
            """,
            state["investigation_id"],
            state["drift_event_id"],
            last_completed_node,
            _serialize_state(state),
        )
