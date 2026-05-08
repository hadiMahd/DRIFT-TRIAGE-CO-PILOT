"""Durable drift window persistence for the platform service."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass


TABLE_SQL = """
CREATE TABLE IF NOT EXISTS platform_drift_state (
    state_id BOOLEAN PRIMARY KEY DEFAULT TRUE,
    drift_accumulator JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_severity TEXT NOT NULL DEFAULT 'stable',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (state_id = TRUE)
);
"""

SEED_SQL = """
INSERT INTO platform_drift_state (state_id, drift_accumulator, last_severity)
VALUES (TRUE, '[]'::jsonb, 'stable')
ON CONFLICT (state_id) DO NOTHING;
"""


@dataclass
class DriftRuntimeState:
    accumulator: list[dict]
    last_severity: str


class DriftStateStore:
    def __init__(self, postgres_dsn: str, window_size: int) -> None:
        self._postgres_dsn = postgres_dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._window_size = window_size

    @staticmethod
    def _load_asyncpg():
        try:
            return importlib.import_module("asyncpg")
        except ImportError as exc:
            raise RuntimeError(
                "asyncpg is required for drift state persistence. "
                "Install platform dependencies before using DriftStateStore."
            ) from exc

    async def ensure_schema(self) -> None:
        asyncpg = self._load_asyncpg()
        connection = await asyncpg.connect(self._postgres_dsn, timeout=5)
        try:
            await connection.execute(TABLE_SQL)
            await connection.execute(SEED_SQL)
        finally:
            await connection.close()

    async def load_state(self) -> DriftRuntimeState:
        asyncpg = self._load_asyncpg()
        connection = await asyncpg.connect(self._postgres_dsn, timeout=5)
        try:
            await connection.execute(TABLE_SQL)
            await connection.execute(SEED_SQL)
            row = await connection.fetchrow(
                """
                SELECT drift_accumulator, last_severity
                FROM platform_drift_state
                WHERE state_id = TRUE
                """
            )
        finally:
            await connection.close()

        accumulator = []
        if row and row["drift_accumulator"] is not None:
            raw_value = row["drift_accumulator"]
            if isinstance(raw_value, str):
                accumulator = json.loads(raw_value)
            else:
                accumulator = list(raw_value)
        last_severity = row["last_severity"] if row else "stable"
        return DriftRuntimeState(accumulator=accumulator, last_severity=last_severity)

    async def save_state(self, accumulator: list[dict], last_severity: str) -> None:
        capped = accumulator[-self._window_size :]
        asyncpg = self._load_asyncpg()
        connection = await asyncpg.connect(self._postgres_dsn, timeout=5)
        try:
            await connection.execute(
                """
                UPDATE platform_drift_state
                SET drift_accumulator = $1::jsonb,
                    last_severity = $2,
                    updated_at = NOW()
                WHERE state_id = TRUE
                """,
                json.dumps(capped),
                last_severity,
            )
        finally:
            await connection.close()
