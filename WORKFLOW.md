# Drift Triage Co-Pilot Workflow

This file describes the current live workflow in the repo.

## Services

| Service | URL | Role |
| --- | --- | --- |
| Dashboard | `http://localhost:8501` | Operator UI |
| Platform | `http://localhost:8000` | Predictions, drift, registry, queue status |
| Agent | `http://localhost:8001` | Webhook receiver, deterministic triage, HIL routes |
| MLflow | `http://localhost:5000` | Tracking and registry |
| Redis | `localhost:6379` | Queue and DLQ |
| Postgres | `localhost:55432` | HIL approvals, audit, investigations, checkpoints, drift state |
| pgAdmin | `http://localhost:5050` | Postgres browser |

## Startup

```powershell
Copy-Item .env.example .env
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

There is one dashboard flow now:

1. send real prediction traffic to the platform
2. run a real drift report
3. let platform notify agent when severity changes
4. let agent queue safe work
5. let worker finish the slow job
6. review or approve the resulting HIL item when Production could change

## End-To-End Critical Drift Flow

```text
Dashboard -> Platform /predict/
Dashboard -> Platform /drift/report
Platform -> Agent /webhook/drift
Agent -> Redis queue
Worker -> retrain + MLflow candidate registration
Worker -> Agent /hil/notify-candidate
Dashboard -> Agent /hil/pending
Dashboard -> Agent /hil/{id}/approve
Agent -> Platform /registry/promote
Platform -> Postgres audit write -> MLflow alias change -> model reload
```

## Dashboard Buttons

The current dashboard uses:

- `Normal (500)`
- `Moderate Drift`
- `Critical Drift`

These buttons send real prediction traffic and then call `/drift/report`.

There is no separate synthetic "demo alert" button path in the dashboard anymore.

## Severity Behavior

| Severity | Agent Action | Queue? | Approval immediately? |
| --- | --- | --- | --- |
| `stable` | `none` | No | No |
| `moderate` | `replay_test` | Yes | No |
| `critical` | `retrain` | Yes | Not immediately |

Important:

- critical drift does not create an approval instantly
- retraining must finish first
- after retraining, the worker creates a `promote_candidate` HIL approval through the agent

That is why the dashboard tells the operator to wait a few seconds and refresh the HIL inbox.

## Queue And Worker Flow

Queue names:

```text
queue: drift-triage-jobs
DLQ:   DLQ:drift-triage-jobs
```

Expected behavior:

- moderate drift queues `replay_test`
- critical drift queues `retrain`
- worker retraining registers a candidate version in MLflow
- worker resolves the concrete candidate version
- worker calls `POST /hil/notify-candidate`
- a pending `promote_candidate` approval appears in the inbox

Useful checks:

```powershell
docker compose exec redis redis-cli LLEN drift-triage-jobs
docker compose exec redis redis-cli LLEN DLQ:drift-triage-jobs
docker compose logs --tail=120 worker
```

## HIL Approval Flow

The HIL inbox reads:

```text
GET http://localhost:8001/hil/pending
```

Approve/reject uses:

```text
POST http://localhost:8001/hil/{approval_id}/approve
POST http://localhost:8001/hil/{approval_id}/reject
```

Approval behavior:

- approving `promote_candidate` dispatches `POST /registry/promote`
- approving `rollback` dispatches `POST /registry/rollback`
- approvals are persisted in Postgres
- repeated event delivery reuses the same investigation by `drift_event_id`

## Registry Flow

The dashboard registry panel reads:

```text
GET http://localhost:8000/registry/status
GET http://localhost:8000/registry/history
```

Expected status fields:

- `registered_model_name`
- `production_version`
- `candidate_version`
- `previous_production_version`
- `production_metrics`

Expected history behavior:

- every promotion and rollback is written to `promotion_audit`
- rollback uses the previously audited Production version as the operator-visible target

## Presentation Script

1. Open dashboard: `http://localhost:8501`
2. Show service health cards.
3. Click `Moderate Drift` and explain that the agent queues `replay_test`.
4. Click `Critical Drift` and explain that the worker retrains before any approval appears.
5. Wait a few seconds, then click `Refresh` in the HIL inbox.
6. Show the pending `promote_candidate` approval.
7. Approve it.
8. Show `registry/status` with current Production version and metrics.
9. Open promotion history and explain rollback gating.
10. If needed, open pgAdmin or query Postgres to prove audit rows.

## What To Say If Asked Why Critical Does Not Ask For Approval Immediately

Use this explanation:

```text
Critical drift triggers retraining, not an automatic Production change.
Retraining only creates a candidate model version.
Approval is required when we are about to promote that candidate or roll Production back.
That is why the approval appears after the worker finishes retraining, not at the moment drift is detected.
```

## Recovery And Persistence

The repo now proves durable Postgres-backed persistence for:

- HIL approvals
- investigations
- investigation checkpoints
- platform drift state

The optional LangGraph Postgres checkpoint helper still exists, but the live recovery path in this project is the repo-owned Postgres persistence flow.
