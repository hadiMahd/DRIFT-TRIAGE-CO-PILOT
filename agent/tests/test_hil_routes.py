"""Route tests for Human-in-the-Loop approval HTTP endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from agent.app.main import app
from agent.app.schemas.hil_action import HILAction


def build_approval(
    *,
    approval_id: str = "apr-1",
    requested_action: str = "rollback",
    status: str = "pending",
    approved_by: str | None = None,
) -> HILAction:
    """Create a valid HILAction model for route tests."""

    return HILAction(
        approval_id=approval_id,
        investigation_id="inv-1",
        drift_event_id="evt-1",
        requested_action=requested_action,
        target_model_version="7",
        status=status,
        requested_by="agent",
        approved_by=approved_by,
        idempotency_key="rollback:inv-1:7",
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
        updated_at=None,
    )


class HILRouteTests(unittest.TestCase):
    """Validate route behavior without a real Postgres instance."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_pending_returns_approvals(self) -> None:
        approvals = [build_approval(approval_id="apr-1"), build_approval(approval_id="apr-2")]

        with patch(
            "agent.app.routers.hil.request_approval.list_pending_approvals",
            new=AsyncMock(return_value=approvals),
        ):
            response = self.client.get("/hil/pending")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["approvals"]), 2)
        self.assertEqual(body["approvals"][0]["approval_id"], "apr-1")

    def test_get_approval_returns_row(self) -> None:
        with patch(
            "agent.app.routers.hil.request_approval.get_approval",
            new=AsyncMock(return_value=build_approval()),
        ):
            response = self.client.get("/hil/apr-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["approval_id"], "apr-1")

    def test_get_approval_returns_404_if_missing(self) -> None:
        with patch(
            "agent.app.routers.hil.request_approval.get_approval",
            new=AsyncMock(return_value=None),
        ):
            response = self.client.get("/hil/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"]["approval_id"], "missing")

    def test_approve_returns_approved(self) -> None:
        with patch(
            "agent.app.routers.hil.request_approval.get_approval",
            new=AsyncMock(return_value=build_approval()),
        ), patch(
            "agent.app.routers.hil.request_approval.approve_action",
            new=AsyncMock(return_value=build_approval(status="approved", approved_by="jad")),
        ), patch(
            "agent.app.routers.hil._dispatch_rollback",
            new=AsyncMock(return_value=None),
        ):
            response = self.client.post(
                "/hil/apr-1/approve",
                json={"approved_by": "jad", "reason": "Looks safe"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"approval_id": "apr-1", "status": "approved", "message": "Approval approved"},
        )

    def test_reject_returns_rejected(self) -> None:
        with patch(
            "agent.app.routers.hil.request_approval.get_approval",
            new=AsyncMock(return_value=build_approval()),
        ), patch(
            "agent.app.routers.hil.request_approval.reject_action",
            new=AsyncMock(return_value=build_approval(status="rejected", approved_by="jad")),
        ):
            response = self.client.post(
                "/hil/apr-1/reject",
                json={"approved_by": "jad", "reason": "Not safe"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"approval_id": "apr-1", "status": "rejected", "message": "Approval rejected"},
        )

    def test_approve_promote_dispatches_platform_promotion(self) -> None:
        with patch(
            "agent.app.routers.hil.request_approval.get_approval",
            new=AsyncMock(return_value=build_approval(requested_action="promote_candidate")),
        ), patch(
            "agent.app.routers.hil.request_approval.approve_action",
            new=AsyncMock(
                return_value=build_approval(
                    requested_action="promote_candidate",
                    status="approved",
                    approved_by="jad",
                )
            ),
        ), patch(
            "agent.app.routers.hil._resolve_target_version",
            new=AsyncMock(return_value="12"),
        ), patch(
            "agent.app.routers.hil._dispatch_promotion",
            new=AsyncMock(return_value=None),
        ) as dispatch_mock:
            response = self.client.post(
                "/hil/apr-1/approve",
                json={"approved_by": "jad", "reason": "Ship it"},
            )

        self.assertEqual(response.status_code, 200)
        dispatch_mock.assert_awaited_once()

    def test_invalid_transition_returns_409(self) -> None:
        with patch(
            "agent.app.routers.hil.request_approval.get_approval",
            new=AsyncMock(return_value=build_approval()),
        ), patch(
            "agent.app.routers.hil.request_approval.approve_action",
            new=AsyncMock(side_effect=ValueError("Cannot transition approval from 'approved' to 'approved'.")),
        ):
            response = self.client.post(
                "/hil/apr-1/approve",
                json={"approved_by": "jad", "reason": "Already handled"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("Cannot transition approval", response.json()["detail"]["message"])

    def test_service_error_returns_structured_error(self) -> None:
        with patch(
            "agent.app.routers.hil.request_approval.list_pending_approvals",
            new=AsyncMock(side_effect=RuntimeError("approval service unavailable")),
        ):
            response = self.client.get("/hil/pending")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            {"message": "approval service unavailable"},
        )

    def test_health_still_works(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "agent"})


if __name__ == "__main__":
    unittest.main()
