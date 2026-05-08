# Runbook

## Prerequisites
- Docker Desktop running
- `uv` installed for optional local test and bootstrap commands

Windows install for `uv`:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Compose Startup
1. Copy the environment template:
```powershell
Copy-Item .env.example .env
```
2. Keep `LLM_PROVIDER=mock` for the no-key demo.
3. Only fill Azure or LangSmith keys if you explicitly want LLM-backed runs.
4. If `platform/data/model.joblib` is missing, bootstrap it before startup:
```powershell
Copy-Item initial-training/dataset/bank-additional-full.csv platform/data/
cd platform
uv run python -m app.services.run_training
cd ..
```
5. Start the stack:
```powershell
docker compose up --build
```

## URLs
- Dashboard: `http://localhost:8501`
- Platform health: `http://localhost:8000/health`
- Agent health: `http://localhost:8001/health`
- MLflow: `http://localhost:5000`
- Postgres host port: `55432`

## Demo Flow
1. Open the dashboard.
2. Confirm Platform Status and Agent Status are healthy.
3. Click `Run Drift Report`.
4. Confirm the platform webhook reaches the agent.
5. Trigger a demo moderate or critical drift event if you want a queued worker action.
6. Check the Redis queue if needed:
```powershell
docker compose exec redis redis-cli LLEN drift-triage-jobs
```
7. Watch worker logs if needed:
```powershell
docker compose logs -f worker
```
8. Approve or reject pending HIL approvals from the dashboard inbox.

## Notes
- The worker consumes jobs from `drift-triage-jobs`.
- `replay_test` is a stubbed safe handler.
- `retrain` runs the platform training pipeline and registers a candidate in MLflow.
- `rollback` remains a stubbed worker handler and still requires HIL approval before any future production action.
