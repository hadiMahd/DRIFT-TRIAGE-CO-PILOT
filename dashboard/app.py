from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st


st.set_page_config(page_title="Drift Triage Co-Pilot", page_icon="DT", layout="wide")

AGENT = os.getenv("AGENT_BASE_URL", "http://agent:8001").rstrip("/")
PLATFORM = os.getenv("PLATFORM_BASE_URL", "http://platform:8000").rstrip("/")
MLFLOW = os.getenv("MLFLOW_BASE_URL", "http://mlflow:5000").rstrip("/")

PREDICT_PAYLOAD = {
    "age": 40,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191,
}

MODERATE_SHIFT = {**PREDICT_PAYLOAD, "euribor3m": 5.5, "cons_price_idx": 96.0}
CRITICAL_SHIFT = {**PREDICT_PAYLOAD, "euribor3m": 6.5, "cons_price_idx": 102.0}


def _init_state() -> None:
    defaults = {
        "drift_result": None,
        "critical_wait_msg": None,
        "hil_autopoll_until": 0.0,
        "approval_msg": None,
        "rollback_msg": None,
        "last_health": {},
        "last_queue_status": {},
        "last_registry_status": {},
        "last_approvals": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_json(url: str, timeout: float = 5) -> dict[str, Any]:
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(1)
    return {"_error": str(last_error) if last_error else "request failed"}


def _post_json(url: str, body: dict[str, Any], timeout: float = 15) -> dict[str, Any]:
    try:
        response = requests.post(url, json=body, timeout=timeout)
        return response.json()
    except Exception as exc:
        return {"detail": {"message": str(exc)}}


def _predict(payload: dict[str, Any]) -> dict[str, Any]:
    return _post_json(f"{PLATFORM}/predict/", payload, timeout=10)


def _remember_state(key: str, value: Any) -> Any:
    st.session_state[key] = value
    return value


def _health_ok(url: str) -> bool:
    for timeout in (3, 6):
        try:
            if requests.get(f"{url}/health", timeout=timeout).status_code == 200:
                return True
        except Exception:
            continue
    return False


def _health_status(name: str, url: str) -> tuple[bool, bool]:
    ok = _health_ok(url)
    previous = dict(st.session_state.last_health)
    if ok:
        previous[name] = True
        _remember_state("last_health", previous)
        return True, False
    if previous.get(name):
        return True, True
    return False, False


def _resilient_get(url: str, state_key: str, *, default: Any, timeout: float = 5) -> tuple[Any, bool]:
    result = _get_json(url, timeout=timeout)
    if "_error" not in result:
        _remember_state(state_key, result)
        return result, False
    previous = st.session_state.get(state_key, default)
    return previous, True


def _run_drift_demo(label: str, normal_count: int, shift_count: int, shift_payload: dict[str, Any]) -> None:
    total = normal_count + shift_count
    placeholder = st.empty()
    with placeholder.status(f"{label} - sending {total} predictions...") as status:
        for i in range(total):
            _predict(PREDICT_PAYLOAD if i < normal_count else shift_payload)
            if i and i % 100 == 0:
                status.write(f"{i}/{total}")

        status.update(label="Running drift report...")
        result = _get_json(f"{PLATFORM}/drift/report", timeout=30)
        st.session_state.drift_result = result

        if "_error" in result:
            status.update(label=f"Drift report failed: {result['_error']}", state="error")
            return

        report = result.get("report") or {}
        severity = report.get("severity")
        if not severity:
            status.update(label="Drift report failed: invalid response", state="error")
            return

        if result.get("webhook_sent") and severity == "critical":
            st.session_state.hil_autopoll_until = 0.0
            st.session_state.critical_wait_msg = (
                "Retrain was queued. Wait 5-10 seconds, then click Refresh in the HIL inbox."
            )
            status.update(
                label=f"Done - Severity: {severity.upper()}. Retrain queued; refresh the HIL inbox in a few seconds.",
                state="complete",
            )
            return

        if result.get("webhook_sent"):
            status.update(label=f"Done - Severity: {severity.upper()}. Agent notified.", state="complete")
        else:
            webhook_error = str(result.get("webhook_error", "") or "")
            if "ReadTimeout" in webhook_error or "request error" in webhook_error.lower():
                status.update(
                    label=f"Done - Severity: {severity.upper()}. Follow-up is still processing; check the HIL inbox shortly.",
                    state="complete",
                )
                return
            status.update(
                label=f"Done - Severity: {severity.upper()}. {webhook_error}",
                state="complete",
            )


def _render_health() -> None:
    c1, c2, c3, c4 = st.columns(4)
    entries = [
        (c1, "Platform", PLATFORM),
        (c2, "Agent", AGENT),
        (c3, "MLflow", MLFLOW),
        (c4, "Queue / Worker", PLATFORM),
    ]
    for col, name, url in entries:
        ok, stale = _health_status(name, url)
        icon = "OK" if (ok or stale) else "DOWN"
        label = "Connected" if (ok or stale) else "Offline"
        with col:
            st.metric(f"{icon} {name}", label)


def _render_drift_result() -> None:
    if not st.session_state.drift_result:
        st.info("Run a drift scenario to compute the current drift severity.")
        st.caption("Critical drift queues retraining first. Approval appears after the worker finishes retraining.")
        return

    result = st.session_state.drift_result
    if result.get("_error") or "report" not in result:
        st.error(f"Drift report failed: {result.get('_error', 'invalid drift response')}")
        st.caption("Severity is unavailable because the dashboard did not receive a valid /drift/report payload.")
        return

    report = result.get("report") or {}
    severity = report.get("severity")
    if not severity:
        st.error("Drift report failed: severity missing from response.")
        return

    colors = {"stable": "green", "moderate": "orange", "critical": "red"}
    st.metric("Severity", severity, delta_color="off")
    st.markdown(f"**{colors.get(severity, 'grey')}**: `{severity}`")

    psi_scores = report.get("psi_scores") or {}
    if psi_scores:
        summary = ", ".join(f"{k}={v:.2f}" for k, v in psi_scores.items() if v > 0.01)
        if summary:
            st.caption(f"PSI: {summary[:120]}")

    if result.get("webhook_sent"):
        st.success("Webhook sent to agent")
        if severity == "critical":
            st.info("Agent recommended: retrain -> worker trains candidate -> HIL approval appears after retrain completes.")
            if st.session_state.critical_wait_msg:
                st.info(st.session_state.critical_wait_msg)
        elif severity == "moderate":
            st.info("Agent recommended: replay_test -> model metrics verified.")

    webhook_error = str(result.get("webhook_error", "") or "")
    if webhook_error:
        if "ReadTimeout" in webhook_error or "request error" in webhook_error.lower():
            if severity == "critical":
                st.info("Follow-up is still processing. Retrain was triggered; the approval will appear after the worker finishes.")
            else:
                st.info("Follow-up is still processing. The agent was contacted, but the dashboard timed out waiting for the response.")
        else:
            st.info(webhook_error)


def _render_hil_inbox() -> None:
    st.subheader("HIL Approval Inbox")

    ref1, _ref2 = st.columns([1, 4])
    with ref1:
        if st.button("Refresh", key="refresh_inbox", use_container_width=True):
            st.session_state.hil_autopoll_until = 0.0

    if st.session_state.approval_msg:
        msg, kind = st.session_state.approval_msg
        if kind == "success":
            st.success(msg)
        elif kind == "info":
            st.info(msg)
        else:
            st.error(msg)
        st.session_state.approval_msg = None

    approvals_data, approvals_stale = _resilient_get(
        f"{AGENT}/hil/pending",
        "last_approvals",
        default={"approvals": []},
        timeout=10,
    )
    reg_live, registry_stale = _resilient_get(
        f"{PLATFORM}/registry/status",
        "last_registry_status",
        default={},
        timeout=10,
    )
    queue_live, queue_stale = _resilient_get(
        f"{PLATFORM}/queue/status",
        "last_queue_status",
        default={},
        timeout=10,
    )

    approvals = approvals_data.get("approvals") or []
    stale_bits = []
    if approvals_stale:
        stale_bits.append("HIL inbox")
    if queue_stale:
        stale_bits.append("queue")
    if registry_stale:
        stale_bits.append("registry")
    if stale_bits:
        st.caption(f"Using last successful data for: {', '.join(stale_bits)}")

    if not approvals:
        candidate = reg_live.get("candidate_version")
        production = reg_live.get("production_version")
        queue_length = queue_live.get("queue_length")
        dlq_length = queue_live.get("dlq_length")
        if candidate and candidate != production:
            st.warning(
                f"Candidate version {candidate} exists but no pending approval is visible right now. "
                "The worker may still be finishing retrain or the approval was already handled."
            )
        else:
            st.info("No pending approvals yet. After critical drift, wait for worker retraining to finish, then the approval will appear here.")
        if st.session_state.critical_wait_msg:
            st.caption(st.session_state.critical_wait_msg)
        if queue_length is not None:
            st.caption(f"Queue length: {queue_length} | DLQ: {dlq_length}")
        return

    st.session_state.hil_autopoll_until = 0.0

    for approval in approvals:
        with st.container(border=True):
            st.markdown(f"**Action:** `{approval.get('requested_action', '?')}`")
            st.caption(
                f"Target: `{approval.get('target_model_version', '?')}` | "
                f"Status: `{approval.get('status', '?')}`"
            )
            st.caption(f"ID: `{approval['approval_id'][:8]}...`")

            approver = st.text_input("Approver", key=f"name_{approval['approval_id']}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Approve", key=f"approve_{approval['approval_id']}", use_container_width=True):
                    response = _post_json(
                        f"{AGENT}/hil/{approval['approval_id']}/approve",
                        {"approved_by": approver or "demo-user"},
                    )
                    if response.get("status") == "approved":
                        st.session_state.approval_msg = ("Approval completed.", "success")
                    else:
                        detail = response.get("detail", {})
                        st.session_state.approval_msg = (detail.get("message", "Approve failed"), "error")
                    st.rerun()
            with c2:
                if st.button("Reject", key=f"reject_{approval['approval_id']}", use_container_width=True):
                    response = _post_json(
                        f"{AGENT}/hil/{approval['approval_id']}/reject",
                        {"approved_by": approver or "demo-user", "reason": "Manual rejection"},
                    )
                    if response.get("status") == "rejected":
                        st.session_state.approval_msg = ("Approval rejected.", "info")
                    else:
                        detail = response.get("detail", {})
                        st.session_state.approval_msg = (detail.get("message", "Reject failed"), "error")
                    st.rerun()


def _render_registry() -> None:
    st.subheader("Registry Status")
    if st.session_state.rollback_msg:
        msg, kind = st.session_state.rollback_msg
        if kind == "success":
            st.success(msg)
        else:
            st.error(msg)
        st.session_state.rollback_msg = None

    registry, registry_stale = _resilient_get(
        f"{PLATFORM}/registry/status",
        "last_registry_status",
        default={},
    )
    if registry_stale:
        st.caption("Showing last successful registry response.")
    if not registry:
        st.caption("Registry unavailable.")
        return

    production = registry.get("production_version")
    candidate = registry.get("candidate_version")
    metrics = registry.get("production_metrics") or {}

    if production:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Model", registry.get("registered_model_name", "-"))
        with c2:
            st.metric("Version", f"v{production}")
        with c3:
            st.metric("Recall", f"{metrics.get('test_recall', 0):.3f}")
        with c4:
            st.metric("F1", f"{metrics.get('test_f1', 0):.3f}")
        with c5:
            st.metric("AUC", f"{metrics.get('test_roc_auc', 0):.3f}")
        if candidate:
            st.caption(f"Candidate: v{candidate}")
    else:
        st.info("No Production model promoted yet")

    history = _get_json(f"{PLATFORM}/registry/history", timeout=10)
    with st.expander("Promotion History"):
        records = history.get("history") or []
        if not records:
            st.caption("No promotions yet.")
            return
        latest = records[0]
        st.caption(
            f"{latest['timestamp'][:19]} | {latest['from_alias']} -> {latest['to_alias']} | "
            f"{latest['model_uri']} by {latest['approved_by']}"
        )
        previous = latest.get("previous_version")
        if previous:
            approval_id = st.text_input("Rollback approval ID", key="rollback_approval_id")
            approved_by = st.text_input("Rollback approver", key="rollback_approved_by")
            if st.button(f"Rollback to v{previous}", use_container_width=True, disabled=not approval_id.strip()):
                response = _post_json(
                    f"{PLATFORM}/registry/rollback",
                    {
                        "target_version": previous,
                        "approval_id": approval_id.strip(),
                        "approved_by": approved_by.strip() or "admin",
                    },
                )
                if response.get("status") == "ok":
                    st.session_state.rollback_msg = (f"Rolled back to v{previous}", "success")
                else:
                    st.session_state.rollback_msg = (str(response.get("detail", "Rollback failed")), "error")
                st.rerun()


def main() -> None:
    _init_state()

    st.title("Drift Triage Co-Pilot")
    st.caption("Self-healing MLOps stack")

    _render_health()
    st.divider()

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.subheader("Drift Monitoring")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Normal (500)", use_container_width=True):
                _run_drift_demo("Normal", 500, 0, PREDICT_PAYLOAD)
        with c2:
            if st.button("Moderate Drift", use_container_width=True):
                _run_drift_demo("Moderate", 250, 250, MODERATE_SHIFT)
        with c3:
            if st.button("Critical Drift", use_container_width=True):
                _run_drift_demo("Critical", 100, 400, CRITICAL_SHIFT)
        _render_drift_result()

    with right:
        _render_hil_inbox()

    st.divider()
    _render_registry()
    st.caption(f"Last refresh: {time.strftime('%H:%M:%S')} | Platform: {PLATFORM} | Agent: {AGENT}")


if __name__ == "__main__":
    main()
