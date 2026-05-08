# Drift Triage Co-Pilot

Self-healing MLOps stack — ML model serving, drift detection, automated retraining, and human-in-the-loop approvals with full audit trails.

## Quick Start

```bash
cp .env.example .env
docker compose up -d --build
```

All 8 services start with health checks. Open the dashboard at **http://localhost:8501**.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  Dashboard  │     │    Agent    │     │    Worker    │
│   :8501     │     │    :8001    │     │   (no port)  │
│  Streamlit  │     │  LangGraph  │     │  Redis       │
│  Operator   │     │  Triage +   │     │  Consumer    │
│  UI         │     │  HIL Routes │     │              │
└──────┬──────┘     └─────┬───────┘     └──────┬───────┘
       │                  │                    │
       │     ┌────────────┴──────────┐         │
       └─────┤       Platform        ├─────────┘
             │        :8000          │
             │  Predict, Drift,      │
             │  Registry, Rollback   │
             └───────┬───────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────┴────┐ ┌────┴────┐  ┌───┴──────┐
    │ MLflow  │ │Postgres │  │  Redis   │
    │  :5000  │ │ :5432   │  │  :6379   │
    │Registry │ │HIL+Audit│  │Queue+DLQ │
    └─────────┘ └─────────┘  └──────────┘
```

## Services

| Service | Port | Tech | Role |
|---------|------|------|------|
| **Dashboard** | 8501 | Streamlit | Operator UI — health, drift monitoring, HIL inbox, registry, rollback |
| **Platform** | 8000 | FastAPI + sklearn | Model serving, drift detection (PSI/chi²), registry, promotion/rollback |
| **Agent** | 8001 | FastAPI + LangGraph | Drift webhook receiver, deterministic triage, HIL approval API, job dispatch |
| **Worker** | — | Python + redis | Background queue consumer — retrain, replay test, rollback (3-retry + DLQ) |
| **MLflow** | 5000 | MLflow 3.12 | Experiment tracking, model registry with versioned artifacts |
| **Postgres** | 5432 | PostgreSQL 16 | HIL approvals, promotion audit trail, investigations |
| **Redis** | 6379 | Redis 7 | Job queue (`drift-triage-jobs`) + dead-letter queue + idempotency |
| **pgAdmin** | 5050 | pgAdmin 4 | Database browser — connect to `postgres:5432`, user `user`, pass `pass` |

## Key Features

- **Drift detection** — PSI on numeric features, chi-squared on categoricals, output drift on prediction probabilities
- **Deterministic triage** — severity → action mapping is hardcoded (stable→none, moderate→replay, critical→retrain). LLM is only used for text summarization, never for production decisions
- **Human-in-the-loop** — promotion and rollback require approved HIL approvals in Postgres
- **Audit-first** — Postgres audit written before MLflow alias changes. Every promotion/rollback records who, what, when, and the previous version
- **Model hot-reload** — after promote/rollback, the platform reloads the model from MLflow into memory with zero downtime
- **Idempotency** — duplicate webhooks, jobs, and approvals detected at Redis layer (SADD/SETNX) and Postgres layer (UNIQUE constraint)
- **Dead-letter queue** — failed jobs persist in `DLQ:drift-triage-jobs` for debugging
- **Explicit rollback** — `previous_version` column in audit table. Dashboard shows rollback button for the most recent promotion

## API Endpoints

### Platform (8000)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/predict/` | Single prediction, accumulates history for drift |
| GET | `/drift/report` | Compute drift, emit webhook on severity change |
| GET | `/registry/status` | Current Production/Candidate versions + metrics + previous version |
| POST | `/registry/promote` | Promote candidate to Production (requires approved_by) |
| POST | `/registry/rollback` | Rollback to a version (requires approved HIL approval) |
| GET | `/registry/history` | Promotion audit trail |
| GET | `/queue/status` | Redis queue/DLQ lengths |

### Agent (8001)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/webhook/drift` | Receive drift alert, run LangGraph investigation |
| GET | `/hil/pending` | List pending HIL approvals |
| GET | `/hil/{id}` | Get one approval |
| POST | `/hil/{id}/approve` | Approve pending action |
| POST | `/hil/{id}/reject` | Reject pending action |
| POST | `/hil/notify-candidate` | Worker creates approval after retrain |

## Demo Flow

1. Open **http://localhost:8501** — 4 green health cards
2. Click **Critical Drift** — sends 500 predictions (100 normal + 400 shifted)
3. Drift report: severity **critical**, webhook sent to agent
4. Agent triages → **retrain** queued to Redis → Worker picks up → trains new candidate
5. Worker notifies agent → **promote_candidate** approval created
6. HIL Inbox shows candidate with metrics (Recall ≥ 0.75: ✅)
7. Click **Approve** → enter approver name → see "Promotion approved!"
8. Registry Status: Production v26 with metrics, previous version v1
9. Promotion History: most recent promotion with **Rollback to v1** button
10. Open **pgAdmin** at http://localhost:5050 → browse `hil_approvals`, `promotion_audit`

## Useful Commands

```bash
# Service status
docker compose ps

# Logs
docker compose logs --tail=50 worker
docker compose logs --tail=50 platform

# Queue inspection
docker compose exec redis redis-cli LLEN drift-triage-jobs
docker compose exec redis redis-cli LLEN DLQ:drift-triage-jobs

# Postgres inspection
docker compose exec postgres psql -U user -d drift -c "SELECT * FROM promotion_audit ORDER BY id DESC;"
docker compose exec postgres psql -U user -d drift -c "SELECT * FROM hil_approvals ORDER BY created_at DESC;"

# Rebuild a single service
docker compose build platform && docker compose up -d platform
```

## Testing

```bash
# Platform tests (25)
cd platform && uv run pytest tests/ -v

# Agent tests (61)
cd agent && PYTHONPATH=.. uv run pytest tests/ -v

# Worker handler tests (4)
cd platform && PYTHONPATH=. uv run pytest ../worker/app/worker/test_handlers.py -v

# Verify compose syntax
docker compose config --quiet
```

## Project Structure

```
├── platform/           # Model serving, drift detection, registry
│   └── app/
│       ├── main.py     # FastAPI assembly, lifespan
│       ├── routers/    # predict, drift, registry, queue
│       ├── services/   # run_training, validate_promotion, compute_drift
│       └── schemas/    # Pydantic request/response models
├── agent/              # LangGraph triage, HIL approval API
│   └── app/
│       ├── graph/      # StateGraph nodes (triage, action, execute, comms)
│       ├── routers/    # webhook, hil
│       ├── services/   # request_approval, manage_checkpoints
│       └── tools/      # queue_client, dispatch_replay/retrain/rollback
├── worker/             # Redis queue consumer
│   └── app/worker/
│       └── consume_queue.py
├── dashboard/          # Streamlit operator UI
│   └── app.py
├── mlflow/             # MLflow tracking server Dockerfile
├── postgres/           # init.sql — database schema
├── docker-compose.yml  # 8-service orchestration
└── docs/
    ├── EXPLANATION.md  # Full code review guide (every function, every decision)
    ├── WORKFLOW.md     # Scenario workflows, presentation script
    └── ARCH.md         # Architecture overview
```

## Safety Model

**Production is never changed automatically.** Only `POST /registry/promote` and `POST /registry/rollback` can change the Production alias, and both require HIL approval tracked in Postgres.

| Action | Can change Production? | HIL required? |
|--------|----------------------|---------------|
| Retrain (critical drift) | No — creates candidate only | No |
| Replay test | No — read-only validation | No |
| Promote | **Yes** — requires `approved_by` | Yes |
| Rollback | **Yes** — requires approved HIL approval | Yes |

See [EXPLANATION.md](EXPLANATION.md) for a comprehensive 1500-line code review explaining every function, design decision, and architectural pattern.
