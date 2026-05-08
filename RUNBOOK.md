# Runbook

## Prerequisites

- Docker Desktop running
- optional local Python environment for tests
- `.env` copied from `.env.example`

## Startup

```powershell
Copy-Item .env.example .env
docker compose up -d --build
docker compose ps
```

Recommended default:

- keep `LLM_PROVIDER=mock` for the no-key demo path

If `platform/data/model.joblib` is missing, bootstrap it first:

```powershell
Copy-Item initial-training/dataset/bank-additional-full.csv platform/data/
cd platform
uv run python -m app.services.run_training
cd ..
```

## URLs

- Dashboard: `http://localhost:8501`
- Platform health: `http://localhost:8000/health`
- Agent health: `http://localhost:8001/health`
- MLflow: `http://localhost:5000`
- pgAdmin: `http://localhost:5050`
- Postgres host port: `55432`

## Demo Steps

1. Open the dashboard.
2. Confirm Platform, Agent, MLflow, and Queue/Worker are connected.
3. Click one of the drift buttons:
   - `Normal (500)`
   - `Moderate Drift`
   - `Critical Drift`
4. Read the drift result panel.
5. For critical drift:
   - retraining is queued first
   - wait about 5-10 seconds
   - click `Refresh` in the HIL inbox
6. Approve or reject the pending HIL approval.
7. Review registry status and promotion history.
8. Use pgAdmin or `psql` if you need to prove audit rows.

## Live Checks

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8000/registry/status
Invoke-RestMethod http://127.0.0.1:8000/registry/history
Invoke-RestMethod http://127.0.0.1:8000/queue/status
Invoke-RestMethod http://127.0.0.1:8001/hil/pending
```

## Queue And Worker

- Queue: `drift-triage-jobs`
- DLQ: `DLQ:drift-triage-jobs`

Useful commands:

```powershell
docker compose exec redis redis-cli LLEN drift-triage-jobs
docker compose exec redis redis-cli LLEN DLQ:drift-triage-jobs
docker compose logs --tail=120 worker
```

Current behavior:

- `replay_test` is implemented as a safe queued validation path
- `retrain` runs the platform training pipeline and registers a candidate in MLflow
- worker rollback dispatch is implemented and requires `approval_id`
- candidate notification to the agent is retried to avoid losing the HIL approval step

## Audit Checks

```powershell
docker compose exec postgres psql -U user -d drift -c "SELECT * FROM hil_approvals ORDER BY created_at DESC LIMIT 10;"
docker compose exec postgres psql -U user -d drift -c "SELECT * FROM promotion_audit ORDER BY timestamp DESC LIMIT 10;"
docker compose exec postgres psql -U user -d drift -c "SELECT * FROM investigations ORDER BY updated_at DESC LIMIT 10;"
docker compose exec postgres psql -U user -d drift -c "SELECT * FROM investigation_checkpoints ORDER BY updated_at DESC LIMIT 10;"
```

## Troubleshooting

### Critical drift finished but no approval appeared yet

- wait a few seconds for worker retraining to finish
- click `Refresh` in the HIL inbox
- check `docker compose logs --tail=120 worker`
- check `Invoke-RestMethod http://127.0.0.1:8001/hil/pending`

### Dashboard says request timeout

- confirm platform and agent health endpoints return `200`
- refresh the dashboard once with `Ctrl+F5`
- inspect:

```powershell
docker compose logs --tail=80 dashboard
docker compose logs --tail=80 platform
docker compose logs --tail=80 agent
```

### Approval exists but Production did not change

- inspect the approval row in Postgres
- confirm the row is `approved`
- inspect platform logs for promotion or rollback validation errors
- inspect `promotion_audit` to confirm whether the audit write succeeded
