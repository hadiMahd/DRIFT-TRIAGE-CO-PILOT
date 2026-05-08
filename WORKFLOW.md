# Drift Triage Co-Pilot Workflow

This file explains how the full Week 5 app works during the demo. It separates the real monitoring path from the synthetic demo path because they are intentionally different.

## Services

| Service | URL | Role |
| --- | --- | --- |
| Dashboard | http://localhost:8501 | Operator UI for health, drift, queue, registry, and HIL approvals |
| Platform | http://localhost:8000 | Prediction API, drift report, queue status, registry status, promotion gate |
| Agent | http://localhost:8001 | Drift webhook receiver, LangGraph triage, HIL routes, Redis dispatch |
| MLflow | http://localhost:5000 | Tracking server and model registry |
| Redis | localhost:6379 | Queue and DLQ for slow jobs |
| Postgres | localhost:55432 | HIL approvals, investigations, promotion audit |

## Startup

From the repo root:

```powershell
cp .env.example .env
docker compose up -d --build
docker compose ps
```

Expected result:

- `postgres`: healthy
- `redis`: healthy
- `mlflow`: healthy
- `platform`: healthy
- `agent`: healthy
- `dashboard`: healthy
- `worker`: up

## Main Mental Model

There are two different dashboard flows:

1. Real Drift Monitoring
2. Demo Agent Alerts

They are separate on purpose.

## Flow 1: Real Drift Monitoring

This is the real platform path:

```text
Dashboard -> Platform /predict/
Dashboard -> Platform /drift/report
Platform -> Agent /webhook/drift only if drift severity changes
```

The dashboard button `Generate 60 Sample Predictions` sends 60 real requests to:

```text
POST http://localhost:8000/predict/
```

Those predictions fill the platform rolling drift window. The button does not fake drift by itself; it creates real prediction history.

Then `Run Real Drift Report` calls:

```text
GET http://localhost:8000/drift/report
```

The platform computes PSI, categorical drift, and output drift from accumulated prediction history.

Webhook status meanings:

| Status | Meaning |
| --- | --- |
| `waiting_for_data` | Platform does not have enough prediction history yet |
| `suppressed` | Drift report ran, but severity did not change, so no webhook was emitted |
| `sent` | Platform emitted a webhook to the agent |
| `failed` | Platform tried to send a webhook but the agent call failed |

Important: `suppressed` is not a failure. It means the platform did the correct thing and avoided sending duplicate alerts.

## Flow 2: Demo Agent Alerts

This is the synthetic demo path:

```text
Dashboard -> Agent /webhook/drift
```

The buttons `Send Stable Demo Alert`, `Send Moderate Demo Alert`, and `Send Critical Demo Alert (queues retrain)` bypass platform drift history and send a valid `DriftAlert` payload directly to the agent.

This path exists so the presentation can quickly show agent routing without waiting for natural drift changes.

Agent action policy:

| Severity | Agent Action | Approval Required? | Why |
| --- | --- | --- | --- |
| `stable` | `none` | No | Nothing needs to happen |
| `moderate` | `replay_test` | No | Replay is a safe queued check |
| `critical` | `retrain` | No | Retrain creates a candidate model only |
| rollback | `rollback` | Yes | Rollback would affect Production |
| promotion | `promote_candidate` | Yes | Promotion would affect Production |

Critical drift does not ask for approval because retraining does not change Production. It only creates a new candidate model in MLflow.

## Queue And Worker Flow

Safe slow actions are queued in Redis:

```text
Agent -> Redis queue drift-triage-jobs -> Worker -> MLflow/platform action
```

Queue names:

```text
queue: drift-triage-jobs
DLQ:   DLQ:drift-triage-jobs
```

Expected behavior:

- Moderate demo alert queues `replay_test`.
- Worker consumes `replay_test` and logs `replay_complete`.
- Critical demo alert queues `retrain`.
- Worker consumes `retrain`, runs training, and registers a new candidate in MLflow.
- Queue length may return to `0` quickly because the worker consumed the job.
- DLQ can contain rollback safety jobs. In this demo, rollback is intentionally not implemented as an automatic Production mutation.

Useful checks:

```powershell
docker compose exec redis redis-cli LLEN drift-triage-jobs
docker compose exec redis redis-cli LLEN DLQ:drift-triage-jobs
docker compose logs --tail=120 worker
```

## HIL Approval Flow

HIL means human-in-the-loop approval. It is only for Production-changing actions.

The dashboard HIL Inbox reads:

```text
GET http://localhost:8001/hil/pending
```

Approve/reject uses:

```text
POST http://localhost:8001/hil/{approval_id}/approve
POST http://localhost:8001/hil/{approval_id}/reject
```

Critical demo alert will not create a pending approval because it queues retraining only. Retraining creates a candidate model, not a Production change.

The actions that require HIL are:

- rollback
- promote candidate to Production

## Registry And MLflow Flow

The dashboard registry panel reads:

```text
GET http://localhost:8000/registry/status
```

Expected demo state:

- Registered model: `bank_marketing_pipeline`
- Candidate version: present after retrain
- Production version: may be empty/null

This is safe. Retraining creates candidate versions only. Production is not changed automatically.

Open MLflow:

```text
http://localhost:5000
```

Look for:

- model name: `bank_marketing_pipeline`
- experiment/run from training or retraining
- candidate model versions

Ignore MLflow GenAI demo/sample models if they appear.

## Presentation Script

1. Open dashboard: http://localhost:8501
2. Show service health cards.
3. Explain the dashboard has two paths:
   - Real Drift Monitoring uses platform `/predict/` history.
   - Demo Agent Alerts sends synthetic alerts directly to the agent.
4. Click `Generate 60 Sample Predictions`.
5. Show `Prediction Window Readiness`.
6. Click `Run Real Drift Report`.
7. If the status is `suppressed`, explain: the platform ran drift detection, but severity did not change, so it correctly suppressed a duplicate webhook.
8. Click `Send Moderate Demo Alert`.
9. Show action `replay_test`, status `queued`, and queue `drift-triage-jobs`.
10. Show worker logs or queue panel. Queue length `0` can mean the worker consumed the job.
11. Click `Send Critical Demo Alert (queues retrain)`.
12. Show action `retrain`, status `queued`, and approval required `False`.
13. Explain: critical retrain creates a candidate only, so no HIL approval is needed.
14. Open registry panel or MLflow and show the candidate model version.
15. Show HIL Inbox.
16. Explain: HIL approval appears only for rollback or promotion because those can change Production.
17. If a pending approval exists, approve/reject it from the dashboard.

## What To Say If Asked Why Critical Does Not Ask For Approval

Use this exact explanation:

```text
Critical drift means the agent should respond urgently, but it does not mean it can mutate Production.
Our deterministic policy maps critical drift to retrain, and retrain creates a candidate model only.
Production-changing actions are rollback and promotion, and those require HIL approval.
```

## Current Known Caveat

LangGraph StateGraph is implemented and tested. Postgres HIL persistence is implemented. LangGraph Postgres checkpoint integration is prepared, but full checkpoint resume is not the main proven recovery path yet.

Do not claim full checkpoint resume unless it is tested live.
