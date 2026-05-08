"""Tests for the LangGraph StateGraph wrapper."""

from __future__ import annotations

from unittest import IsolatedAsyncioTestCase
import os
from unittest.mock import AsyncMock, patch

from agent.app.config.settings import AgentSettings, get_settings
from agent.app.graph.build_graph import (
    build_agent_graph,
    configure_langsmith_environment,
    run_investigation,
)
from agent.app.schemas.drift_alert import DriftAlert


def sample_alert(*, severity: str) -> DriftAlert:
    return DriftAlert.model_validate(
        {
            "schema_version": "v1",
            "event_id": f"drift-{severity}-001",
            "created_at": "2026-05-05T12:00:00Z",
            "model_name": "bank_marketing_pipeline",
            "model_version": "1",
            "model_alias": "candidate",
            "severity": severity,
            "window": {
                "size": 200,
                "start": "2026-05-05T11:00:00Z",
                "end": "2026-05-05T12:00:00Z",
            },
            "numeric_drift": [
                {
                    "feature": "euribor3m",
                    "psi": 0.31 if severity == "critical" else 0.12,
                    "severity": severity,
                }
            ],
            "categorical_drift": [],
            "output_drift": {
                "psi": 0.18 if severity == "critical" else 0.08,
                "positive_rate_reference": 0.11,
                "positive_rate_current": 0.22,
                "severity": severity,
            },
        }
    )


def build_settings(**overrides) -> AgentSettings:
    values = {
        "POSTGRES_DSN": "postgresql://user:pass@localhost:5432/drift",
        "REDIS_URL": "redis://localhost:6379/0",
        "PLATFORM_BASE_URL": "http://localhost:8000",
        "LLM_PROVIDER": "mock",
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
        "LANGSMITH_API_KEY": None,
        "LANGSMITH_PROJECT": "drift-triage-copilot",
        "LANGGRAPH_API_KEY": None,
    }
    values.update(overrides)
    return AgentSettings(**values)


class LangGraphWrapperTests(IsolatedAsyncioTestCase):
    """Validate graph compilation and deterministic behavior in mock mode."""

    def setUp(self) -> None:
        self._env_patcher = patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "mock",
                "LANGSMITH_TRACING": "false",
            },
            clear=False,
        )
        self._env_patcher.start()
        get_settings.cache_clear()

    def tearDown(self) -> None:
        self._env_patcher.stop()
        get_settings.cache_clear()

    def test_configure_langsmith_environment_bridges_settings_to_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "agent.app.graph.build_graph.get_settings",
            return_value=build_settings(
                LANGSMITH_TRACING="true",
                LANGSMITH_API_KEY="test-key",
                LANGSMITH_PROJECT="test-project",
                LANGGRAPH_API_KEY="test-langgraph-key",
            ),
        ):
            configure_langsmith_environment()

            self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
            self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "true")
            self.assertEqual(os.environ["LANGSMITH_PROJECT"], "test-project")
            self.assertEqual(os.environ["LANGCHAIN_PROJECT"], "test-project")
            self.assertIn("LANGSMITH_API_KEY", os.environ)
            self.assertIn("LANGGRAPH_API_KEY", os.environ)

    def test_build_agent_graph_returns_invokable_graph(self) -> None:
        graph = build_agent_graph()

        self.assertTrue(callable(getattr(graph, "ainvoke", None)))

    async def test_run_investigation_stable_in_mock_mode(self) -> None:
        state = await run_investigation(sample_alert(severity="stable"))

        self.assertEqual(state["recommended_action"], "none")
        self.assertEqual(state["status"], "resolved")
        self.assertFalse(state["requires_approval"])

    async def test_run_investigation_moderate_in_mock_mode(self) -> None:
        with patch(
            "agent.app.graph.run_execute_action.dispatch_replay_test",
            new=AsyncMock(
                return_value={
                    "job_id": "job-replay",
                    "queued": True,
                    "queue_name": "drift-triage-jobs",
                }
            ),
        ):
            state = await run_investigation(sample_alert(severity="moderate"))

        self.assertEqual(state["recommended_action"], "replay_test")
        self.assertEqual(state["status"], "queued")
        self.assertEqual(state["job_id"], "job-replay")

    async def test_run_investigation_critical_in_mock_mode(self) -> None:
        with patch(
            "agent.app.graph.run_execute_action.dispatch_retrain",
            new=AsyncMock(
                return_value={
                    "job_id": "job-retrain",
                    "queued": True,
                    "queue_name": "drift-triage-jobs",
                }
            ),
        ):
            state = await run_investigation(sample_alert(severity="critical"))

        self.assertEqual(state["recommended_action"], "retrain")
        self.assertEqual(state["status"], "queued")
        self.assertEqual(state["job_id"], "job-retrain")

    async def test_llm_failure_falls_back_to_deterministic_summary(self) -> None:
        with patch(
            "agent.app.graph.run_triage.complete_json",
            side_effect=RuntimeError("llm down"),
        ):
            state = await run_investigation(sample_alert(severity="stable"))

        self.assertEqual(state["recommended_action"], "none")
        self.assertIn("Received stable drift alert", state["comms_summary"])


if __name__ == "__main__":
    import unittest

    unittest.main()
