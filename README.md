# Drift Triage Co-Pilot

Self-healing MLOps stack for model serving, drift detection, deterministic triage, queued remediation, and human-approved Production changes.

## Quick Start

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

Open:

- Dashboard: `http://localhost:8501`
- Platform: `http://localhost:8000/health`
- Agent: `http://localhost:8001/health`
- MLflow: `http://localhost:5000`
- pgAdmin: `http://localhost:5050`

## Services

| Service | Port | Role |
| --- | --- | --- |
| Dashboard | 8501 | Operator UI for drift, queue, registry, HIL approvals, rollback |
| Platform | 8000 | Prediction API, drift report, registry status, promotion, rollback |
| Agent | 8001 | Drift webhook receiver, deterministic LangGraph flow, HIL approval API |
| Worker | - | Redis consumer for retrain, replay test, rollback dispatch |
| MLflow | 5000 | Tracking server and model registry |
| Postgres | 55432 -> 5432 | Audit, HIL approvals, investigations, checkpoints, drift state |
| Redis | 6379 | Queue, DLQ, idempotency |
| pgAdmin | 5050 | Postgres browser |

## Current Behavior

- Drift is computed from real prediction traffic through `/predict/` and `/drift/report`.
- Critical drift queues retraining. It does not change Production directly.
- After retraining, the worker registers a candidate version in MLflow and asks the agent to create a `promote_candidate` HIL approval.
- Approving that HIL item dispatches platform promotion automatically.
- Rollback is approval-gated and writes Postgres audit rows before alias mutation.
- Registry status exposes:
  - registered model name
  - production version
  - candidate version
  - previous production version
  - production metrics

## Main Endpoints

### Platform

| Method | Path |
| --- | --- |
| GET | `/health` |
| POST | `/predict/` |
| GET | `/drift/report` |
| GET | `/registry/status` |
| GET | `/registry/history` |
| POST | `/registry/promote` |
| POST | `/registry/rollback` |
| GET | `/queue/status` |

### Agent

| Method | Path |
| --- | --- |
| GET | `/health` |
| POST | `/webhook/drift` |
| GET | `/hil/pending` |
| GET | `/hil/{approval_id}` |
| POST | `/hil/{approval_id}/approve` |
| POST | `/hil/{approval_id}/reject` |
| POST | `/hil/notify-candidate` |

## Dashboard Flow

1. Open the dashboard.
2. Use one of:
   - `Normal (500)`
   - `Moderate Drift`
   - `Critical Drift`
3. Review the drift result.
4. For critical drift, wait a few seconds for worker retraining to finish, then click `Refresh` in the HIL inbox.
5. Approve or reject the pending HIL item.
6. Check registry status and promotion history.

## Safety Model

Production is never changed automatically by drift detection alone.

| Action | Changes Production? | Approval required? |
| --- | --- | --- |
| Retrain | No | No |
| Replay test | No | No |
| Promote candidate | Yes | Yes |
| Rollback | Yes | Yes |

Promotion and rollback are audit-first: Postgres is written before MLflow alias mutation.

## Testing

```powershell
cd platform
.\.venv\Scripts\python.exe -m pytest tests -v -p no:cacheprovider
cd ..

.\platform\.venv\Scripts\python.exe -m pytest worker\app\worker\test_handlers.py -v -p no:cacheprovider
python -m pytest agent\tests -v
python -m py_compile dashboard\app.py
docker compose config --quiet
```

## Useful Commands

```powershell
docker compose ps
docker compose logs --tail=80 platform
docker compose logs --tail=80 agent
docker compose logs --tail=80 worker

docker compose exec redis redis-cli LLEN drift-triage-jobs
docker compose exec redis redis-cli LLEN DLQ:drift-triage-jobs

docker compose exec postgres psql -U user -d drift -c "SELECT * FROM hil_approvals ORDER BY created_at DESC LIMIT 10;"
docker compose exec postgres psql -U user -d drift -c "SELECT * FROM promotion_audit ORDER BY timestamp DESC LIMIT 10;"
```

## Project Structure

```text
platform/   - serving, drift, registry
agent/      - deterministic investigation flow, HIL routes
worker/     - queue consumer
dashboard/  - Streamlit operator UI
mlflow/     - MLflow container
postgres/   - schema bootstrap
```

## Docs

- [DECISIONS.md](DECISIONS.md)
- [EXPLANATION.md](EXPLANATION.md)
- [RUNBOOK.md](RUNBOOK.md)
- [WORKFLOW.md](WORKFLOW.md)
