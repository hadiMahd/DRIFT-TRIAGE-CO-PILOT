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

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in [
    ("approval_action", None), ("approval_msg", None),
    ("drift_result", None), ("predict_count", 0),
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

# ── Registry Status ───────────────────────────────────────────────────────────

reg = _get(f"{PLATFORM}/registry/status")
r1, r2, r3, r4 = st.columns(4)
if reg:
    with r1:
        st.metric("Model", reg.get("registered_model_name", "—"))
    with r2:
        st.metric("Production", reg.get("production_version") or "—")
    with r3:
        st.metric("Candidate", reg.get("candidate_version") or "—")
    with r4:
        st.metric("Status", reg.get("status", "—"))

st.divider()

# ── Main Area: Drift Monitoring (left) + Human Decision (right) ─────────────

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
    st.subheader("Human Decision")

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
                st.caption(f"Created: `{str(a.get('created_at','?'))[:19].replace('T',' ')}` | Investigation: `{a.get('investigation_id','?')[:8]}...`")

                # Show drifted features from last drift result
                drift = st.session_state.drift_result or {}
                report = drift.get("report", {})
                if report.get("psi_scores"):
                    drifted = [f for f, v in report["psi_scores"].items() if v > 0.01]
                    if drifted:
                        st.caption(f"📊 Drifted features: {', '.join(drifted)}")
                st.caption("Agent detected critical drift → recommended retrain → candidate model ready for review.")

                # Candidate vs Production comparison
                reg = _get(f"{PLATFORM}/registry/status")
                if reg.get("candidate_version"):
                    c1m, c2m, c3m = st.columns(3)
                    with c1m:
                        st.metric("Candidate", reg.get("candidate_version"))
                    with c2m:
                        st.metric("Production", reg.get("production_version") or "none")
                    with c3m:
                        st.metric("Recall ≥ 0.75", "✅" if reg else "?")

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
                        st.rerun()
                with ac2:
                    if st.button("Reject", key=f"rej_{a['approval_id']}", use_container_width=True):
                        body = {"approved_by": approved_by or "demo-user", "reason": "Manual rejection"}
                        r = _post(f"{AGENT}/hil/{a['approval_id']}/reject", body)
                        if r.get("status") == "rejected":
                            st.session_state.approval_msg = ("Promotion rejected.", "info")
                        else:
                            st.session_state.approval_msg = (r.get("detail", {}).get("message", "Reject failed"), "error")
                        st.rerun()

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(f"Last refresh: {time.strftime('%H:%M:%S')} | Platform: {PLATFORM} | Agent: {AGENT}")
