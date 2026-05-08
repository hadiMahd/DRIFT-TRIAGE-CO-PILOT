"""GET /drift/report + internal webhook emission.

GET /report — computes PSI/chi2 from in-memory accumulator, compares
to last severity, and auto-emits webhook to agent on severity change.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import numpy as np
from fastapi import APIRouter, Depends, Request

from app.dependencies import get_http_client, get_settings
from app.schemas.drift_report import DriftReport

router = APIRouter()

NUMERIC_FEATURES = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "day_of_week", "poutcome",
]


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — compares two numeric distributions."""
    if len(reference) < 10 or len(current) < 10:
        return 0.0
    try:
        ref_counts, edges = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=edges)
        ref_counts = ref_counts.astype(np.float64) + 1e-10
        cur_counts = cur_counts.astype(np.float64) + 1e-10
        ref_props = ref_counts / ref_counts.sum()
        cur_props = cur_counts / cur_counts.sum()
        psi = np.sum((ref_props - cur_props) * np.log((ref_props + 1e-10) / (cur_props + 1e-10)))
        return float(np.clip(psi, 0, None))
    except Exception:
        return 0.0


def compute_chi2(reference: np.ndarray, current: np.ndarray) -> float:
    """Chi-squared test for categorical feature drift."""
    if len(reference) < 10 or len(current) < 10:
        return 0.0
    try:
        all_cats = sorted(set(list(reference) + list(current)))
        if len(all_cats) < 2:
            return 0.0
        ref_counts = np.array([np.sum(reference == c) for c in all_cats]).astype(float) + 0.5
        cur_counts = np.array([np.sum(current == c) for c in all_cats]).astype(float) + 0.5
        total = ref_counts.sum() + cur_counts.sum()
        ref_exp = (ref_counts + cur_counts) * ref_counts.sum() / total
        cur_exp = (ref_counts + cur_counts) * cur_counts.sum() / total
        chi2 = np.sum((ref_counts - ref_exp) ** 2 / (ref_exp + 1e-10)) + \
               np.sum((cur_counts - cur_exp) ** 2 / (cur_exp + 1e-10))
        return float(chi2)
    except Exception:
        return 0.0


def classify_drift_value(value: float, moderate: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= moderate:
        return "moderate"
    return "stable"


def drift_report_to_alert(report: DriftReport, settings) -> dict:
    """Convert the internal platform drift report to the agent DriftAlert shape."""
    created_at = report.timestamp.astimezone(timezone.utc)
    event_id = f"drift-{created_at.strftime('%Y%m%dT%H%M%S%fZ')}"

    numeric_drift = [
        {"feature": feature, "psi": value, "severity": classify_drift_value(
            value, settings.drift_severity_moderate, settings.drift_severity_critical)}
        for feature, value in report.psi_scores.items()
    ]
    categorical_drift = [
        {"feature": feature, "p_value": value, "severity": classify_drift_value(
            value, settings.drift_severity_moderate, settings.drift_severity_critical)}
        for feature, value in report.chi2_scores.items()
    ]

    return {
        "schema_version": "v1",
        "event_id": event_id,
        "created_at": created_at.isoformat(),
        "model_name": settings.registered_model_name,
        "model_version": None,
        "model_alias": None,
        "model_uri": None,
        "previous_severity": None,
        "severity": report.severity,
        "window": {"size": settings.drift_window_size, "start": None, "end": created_at.isoformat()},
        "numeric_drift": numeric_drift,
        "categorical_drift": categorical_drift,
        "output_drift": {"psi": report.output_drift, "positive_rate_reference": None,
                          "positive_rate_current": None, "severity": report.severity},
        "idempotency_key": f"{event_id}:{report.severity}",
    }


async def emit_webhook(
    report: DriftReport, client: httpx.AsyncClient, settings,
) -> tuple[bool, str | None, dict | None]:
    payload = drift_report_to_alert(report, settings)
    try:
        response = await client.post(
            f"{settings.agent_base_url}/webhook/drift", json=payload, timeout=10.0,
        )
        if response.status_code == 200:
            try:
                body = response.json()
            except ValueError:
                body = {"message": "agent returned non-JSON"}
            return True, None, body
        return False, f"agent returned {response.status_code}", None
    except httpx.RequestError as exc:
        return False, f"request error: {exc.__class__.__name__}", None


@router.get("/report")
async def get_report(
    request: Request,
    client: httpx.AsyncClient = Depends(get_http_client),
    settings=Depends(get_settings),
) -> dict:
    """Compute drift from accumulator, compare severity, auto-emit webhook."""
    app_state = request.app.state
    accumulator: list[dict] = getattr(app_state, "drift_accumulator", None) or []

    if len(accumulator) < 50:
        report = DriftReport(
            severity="stable", psi_scores={}, chi2_scores={},
            output_drift=0.0, timestamp=datetime.now(timezone.utc),
        )
        success = False
        error = "insufficient data (<50 predictions)"
        webhook_response = None
    else:
        half = len(accumulator) // 2
        ref_data = accumulator[:half]
        cur_data = accumulator[half:]

        psi_scores: dict[str, float] = {}
        for feat in NUMERIC_FEATURES:
            ref_vals = np.array([r[feat] for r in ref_data if feat in r], dtype=np.float64)
            cur_vals = np.array([r[feat] for r in cur_data if feat in r], dtype=np.float64)
            psi_scores[feat] = compute_psi(ref_vals, cur_vals)

        chi2_scores: dict[str, float] = {}
        for feat in CATEGORICAL_FEATURES:
            ref_vals = np.array([r.get(feat, "") for r in ref_data])
            cur_vals = np.array([r.get(feat, "") for r in cur_data])
            chi2_scores[feat] = compute_chi2(ref_vals, cur_vals)

        ref_probas = np.array([r.get("proba", 0) for r in ref_data])
        cur_probas = np.array([r.get("proba", 0) for r in cur_data])
        output_drift = compute_psi(ref_probas, cur_probas)

        max_psi = max(psi_scores.values()) if psi_scores else 0.0
        max_chi2 = max(chi2_scores.values()) if chi2_scores else 0.0
        worst = max(max_psi, max_chi2, output_drift)
        mod = settings.drift_severity_moderate
        crit = settings.drift_severity_critical

        if worst >= crit:
            severity = "critical"
        elif worst >= mod:
            severity = "moderate"
        else:
            severity = "stable"

        report = DriftReport(
            severity=severity,
            psi_scores=psi_scores,
            chi2_scores=chi2_scores,
            output_drift=output_drift,
            timestamp=datetime.now(timezone.utc),
        )

        last = getattr(app_state, "last_severity", "stable")
        if severity != last:
            success, error, webhook_response = await emit_webhook(report, client, settings)
            if success:
                app_state.last_severity = severity
                store = getattr(app_state, "drift_state_store", None)
                if store is not None:
                    try:
                        await store.save_state(accumulator, app_state.last_severity)
                    except Exception:
                        pass
        else:
            success = False
            error = "severity unchanged — webhook suppressed"
            webhook_response = None

    return {
        "report": report.model_dump(mode="json"),
        "webhook_sent": success,
        "webhook_error": error,
        "webhook_response": webhook_response,
    }
