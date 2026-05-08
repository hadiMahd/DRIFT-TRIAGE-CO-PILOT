"""Streamlit dashboard for Drift Triage Co-Pilot — clean, user-friendly layout."""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="Drift Triage Co-Pilot", page_icon="🛰️", layout="wide")

AGENT = os.getenv("AGENT_BASE_URL", "http://agent:8001").rstrip("/")
PLATFORM = os.getenv("PLATFORM_BASE_URL", "http://platform:8000").rstrip("/")

PREDICT_PAYLOAD = {
    "age": 40, "job": "admin.", "marital": "married", "education": "university.degree",
    "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon", "campaign": 1, "pdays": 999,
    "previous": 0, "poutcome": "nonexistent", "emp_var_rate": 1.1,
    "cons_price_idx": 93.994, "cons_conf_idx": -36.4, "euribor3m": 4.857,
    "nr_employed": 5191,
}

SHIFTED_PAYLOAD = {**PREDICT_PAYLOAD, "euribor3m": 5.0, "cons_price_idx": 100.0}

MODERATE_SHIFT = {**PREDICT_PAYLOAD, "euribor3m": 5.5, "cons_price_idx": 96.0}
CRITICAL_SHIFT = {**PREDICT_PAYLOAD, "euribor3m": 6.5, "cons_price_idx": 102.0}


@st.cache_data(ttl=5)
def _get(url: str) -> dict:
    try:
        return requests.get(url, timeout=5).json()
    except Exception:
        return {}


@st.cache_data(ttl=5)
def _post(url: str, body: dict) -> dict:
    try:
        return requests.post(url, json=body, timeout=10).json()
    except Exception:
        return {}


def _health_ok(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except Exception:
        return False


def _predict(payload: dict) -> dict:
    try:
        return requests.post(f"{PLATFORM}/predict/", json=payload, timeout=5).json()
    except Exception:
        return {}


def _mlflow_metrics() -> dict:
    """Fetch latest MLflow run metrics via REST API."""
    try:
        url = f"{os.getenv('MLFLOW_BASE_URL', 'http://mlflow:5000')}/api/2.0/mlflow/runs/search"
        resp = requests.post(url, json={
            "experiment_ids": ["1"],
            "max_results": 1,
            "order_by": ["start_time DESC"],
        }, timeout=5)
        runs = resp.json().get("runs", [])
        if runs:
            raw = runs[0].get("data", {}).get("metrics", [])
            return {m["key"]: m["value"] for m in raw if isinstance(m, dict)}
    except Exception:
        pass
    return {}

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in [
    ("approval_action", None), ("approval_msg", None),
    ("drift_result", None), ("predict_count", 0),
    ("rollback_msg", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Drift Triage Co-Pilot")
st.caption("Self-healing MLOps stack — model serving, drift detection, automated retraining")

# ── Health Bar ────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
for col, name, url in [
    (c1, "Platform", PLATFORM), (c2, "Agent", AGENT),
    (c3, "MLflow", os.getenv("MLFLOW_BASE_URL", "http://mlflow:5000")), (c4, "Queue / Worker", PLATFORM),
]:
    ok = _health_ok(url)
    emoji = "✅" if ok else "❌"
    label = "Connected" if ok else "Offline"
    with col:
        st.metric(f"{emoji} {name}", label)

st.divider()

# ── Main Area: Drift Monitoring (left) + HIL Inbox (right) ───────────────────

left, right = st.columns([1.3, 1], gap="large")

with left:
    st.subheader("Drift Monitoring")

    def _run_drift_demo(label: str, normal_count: int, shift_count: int, shift_payload: dict):
        """Send predictions then auto-run drift report."""
        total = normal_count + shift_count
        placeholder = st.empty()
        with placeholder.status(f"{label} — sending {total} predictions...") as status:
            for i in range(total):
                _predict(PREDICT_PAYLOAD if i < normal_count else shift_payload)
                if i % 100 == 0:
                    status.write(f"{i}/{total}")
            status.update(label="Running drift report...")
            result = _get(f"{PLATFORM}/drift/report")
            st.session_state.drift_result = result
            sev = result.get("report", {}).get("severity", "?")
            if result.get("webhook_sent"):
                status.update(label=f"Done — Severity: {sev.upper()}. Agent notified.", state="complete")
            else:
                status.update(label=f"Done — Severity: {sev}. {result.get('webhook_error', '')}", state="complete")

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        if st.button("Normal (500)", help="All identical — stable", use_container_width=True):
            _run_drift_demo("Normal", 500, 0, PREDICT_PAYLOAD)
    with bc2:
        if st.button("Moderate Drift", help="250 normal + 250 shifted — moderate", use_container_width=True):
            _run_drift_demo("Moderate", 250, 250, MODERATE_SHIFT)
    with bc3:
        if st.button("Critical Drift", help="100 normal + 400 shifted — critical → retrain", use_container_width=True):
            _run_drift_demo("Critical", 100, 400, CRITICAL_SHIFT)

    if st.session_state.drift_result:
        d = st.session_state.drift_result
        report = d.get("report", {})
        sev = report.get("severity", "unknown")
        sev_color = {"stable": "green", "moderate": "orange", "critical": "red"}.get(sev, "grey")
        st.metric("Severity", sev, delta_color="off")
        st.markdown(f"**{sev_color}**: `{sev}`")
        if report.get("psi_scores"):
            st.caption(f"PSI: {', '.join(f'{k}={v:.2f}' for k,v in report['psi_scores'].items() if v > 0.01)[:120]}")
        if d.get("webhook_sent"):
            st.success("Webhook sent to agent")
            if sev == "critical":
                st.info("Agent recommended: **retrain** → worker picks up → candidate registered → HIL inbox populates")
            elif sev == "moderate":
                st.info("Agent recommended: **replay_test** → model metrics verified")
            st.info(d["webhook_error"])
    else:
        st.info("Click 'Fill Window' then 'Run Drift Report' to see drift detection in action.")
        st.caption("Tip: 400 normal + 100 shifted predictions triggers PSI on euribor3m")

with right:
    st.subheader("HIL Approval Inbox")

    ref1, ref2 = st.columns([1, 4])
    with ref1:
        if st.button("Refresh", key="refresh_inbox", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if st.session_state.approval_msg:
        msg, kind = st.session_state.approval_msg
        if kind == "success":
            st.success(msg)
        else:
            st.error(msg)
        st.session_state.approval_msg = None

    approvals_data = _get(f"{AGENT}/hil/pending")
    approvals = (approvals_data.get("approvals") or [] if isinstance(approvals_data, dict) else [])

    if not approvals:
        st.info("No pending approvals — trigger a critical drift to auto-generate one.")
    else:
        for a in approvals:
            with st.container(border=True):
                st.markdown(f"**Action:** `{a.get('requested_action','?')}`")
                st.caption(f"Target: `{a.get('target_model_version','?')}` | Status: `{a.get('status','?')}`")
                st.caption(f"ID: `{a['approval_id'][:8]}...`")

                # Candidate vs Production comparison with actual metrics
                reg = _get(f"{PLATFORM}/registry/status")
                mlflow = _mlflow_metrics()
                recall = mlflow.get("test_recall", 0)
                f1 = mlflow.get("test_f1", 0)
                auc = mlflow.get("test_roc_auc", 0)
                threshold = mlflow.get("operating_threshold", 0)

                if recall:
                    c1m, c2m, c3m, c4m = st.columns(4)
                    with c1m:
                        st.metric("Candidate", reg.get("candidate_version", "?") or "—")
                    with c2m:
                        st.metric("Recall", f"{recall:.3f}", delta=f">= 0.75: {'✅' if recall >= 0.75 else '❌'}")
                    with c3m:
                        st.metric("F1", f"{f1:.3f}")
                    with c4m:
                        st.metric("AUC", f"{auc:.3f}")
                    if threshold:
                        st.caption(f"Operating threshold: {threshold:.4f}")

                ac1, ac2 = st.columns(2)
                with ac1:
                    approved_by = st.text_input("Approver", key=f"name_{a['approval_id']}")
                    if st.button("Approve", key=f"apr_{a['approval_id']}", use_container_width=True):
                        body = {"approved_by": approved_by or "demo-user"}
                        r = _post(f"{AGENT}/hil/{a['approval_id']}/approve", body)
                        if r.get("status") == "approved":
                            st.session_state.approval_msg = ("Promotion approved!", "success")
                        else:
                            st.session_state.approval_msg = (r.get("detail", {}).get("message", "Approve failed"), "error")
                        st.cache_data.clear()
                        st.rerun()
                with ac2:
                    if st.button("Reject", key=f"rej_{a['approval_id']}", use_container_width=True):
                        body = {"approved_by": approved_by or "demo-user", "reason": "Manual rejection"}
                        r = _post(f"{AGENT}/hil/{a['approval_id']}/reject", body)
                        if r.get("status") == "rejected":
                            st.session_state.approval_msg = ("Promotion rejected.", "info")
                        else:
                            st.session_state.approval_msg = (r.get("detail", {}).get("message", "Reject failed"), "error")
                        st.cache_data.clear()
                        st.rerun()

st.divider()

# ── Registry Status ───────────────────────────────────────────────────────────

st.subheader("Registry Status")

if st.session_state.rollback_msg:
    msg, kind = st.session_state.rollback_msg
    if kind == "success":
        st.success(msg)
    else:
        st.error(msg)
    st.session_state.rollback_msg = None

reg = _get(f"{PLATFORM}/registry/status")
if reg:
    prod_ver = reg.get("production_version")
    prod_metrics = reg.get("production_metrics") or {}
    cand_ver = reg.get("candidate_version")

    if prod_ver:
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        with rc1:
            st.metric("Model", reg.get("registered_model_name", "—"))
        with rc2:
            st.metric("Version", f"v{prod_ver}")
        with rc3:
            st.metric("Recall", f"{prod_metrics.get('test_recall', 0):.3f}")
        with rc4:
            st.metric("F1", f"{prod_metrics.get('test_f1', 0):.3f}")
        with rc5:
            st.metric("AUC", f"{prod_metrics.get('test_roc_auc', 0):.3f}")

        if cand_ver:
            st.caption(f"Candidate: v{cand_ver} — pending approval")
    else:
        st.info("No Production model promoted yet")

    # Promotion history with rollback
    with st.expander("Promotion History"):
        try:
            hist = requests.get(f"{PLATFORM}/registry/history", timeout=5).json()
            records = hist.get("history", [])
            if records:
                rec = records[0]
                ver = rec["model_uri"].split("/")[-1] if "/" in rec["model_uri"] else rec["model_uri"]
                prev = rec.get("previous_version")
                st.caption(
                    f"{rec['timestamp'][:19]} | {rec['from_alias']} → {rec['to_alias']} | "
                    f"v{ver} by {rec['approved_by']}"
                )

                if prev:
                    rollback_approval_id = st.text_input("Rollback approval ID", key="rollback_approval_id")
                    rollback_approved_by = st.text_input("Rollback approver", key="rollback_approved_by")
                    rollback_ready = bool(rollback_approval_id.strip())
                    bc1, bc2 = st.columns([1, 4])
                    with bc1:
                        if st.button(
                            f"Rollback to v{prev}",
                            key="rollback_btn",
                            use_container_width=True,
                            disabled=not rollback_ready,
                        ):
                            r = _post(
                                f"{PLATFORM}/registry/rollback",
                                {
                                    "target_version": prev,
                                    "approval_id": rollback_approval_id.strip(),
                                    "approved_by": rollback_approved_by.strip() or "admin",
                                },
                            )
                            if r.get("status") == "ok":
                                st.session_state.rollback_msg = (f"Rolled back to v{prev}", "success")
                            else:
                                st.session_state.rollback_msg = (r.get("detail", "Rollback failed"), "error")
                            st.cache_data.clear()
                            st.rerun()
                    with bc2:
                        st.caption(f"Rollback target: v{prev}")
                        if not rollback_ready:
                            st.caption("Requires an approved HIL rollback approval ID.")
            else:
                st.caption("No promotions yet.")
        except Exception:
            st.caption("History unavailable.")
else:
    st.caption("Registry unavailable")

# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(f"Last refresh: {time.strftime('%H:%M:%S')} | Platform: {PLATFORM} | Agent: {AGENT}")
