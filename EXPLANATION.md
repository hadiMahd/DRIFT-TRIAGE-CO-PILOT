# EXPLANATION.md — Full Code Review Guide

This document explains every design decision, every function, every technology choice, and every architectural pattern in the Drift Triage Co-Pilot project. It is written for a code review where the instructor will ask: "Why did you do it this way?"

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Architecture: Why 8 Microservices](#2-architecture-why-8-microservices)
3. [Technology Choices](#3-technology-choices)
4. [Service Deep Dive: Platform (port 8000)](#4-platform-service-port-8000)
5. [Service Deep Dive: Agent (port 8001)](#5-agent-service-port-8001)
6. [Service Deep Dive: Worker](#6-worker-service-no-port)
7. [Service Deep Dive: Dashboard (port 8501)](#7-dashboard-service-port-8501)
8. [Service Deep Dive: MLflow (port 5000)](#8-mlflow-service-port-5000)
9. [Service Deep Dive: Postgres + pgAdmin](#9-postgres--pgadmin)
10. [Service Deep Dive: Redis (port 6379)](#10-redis-service-port-6379)
11. [Cross-Cutting Design Decisions](#11-cross-cutting-design-decisions)
12. [End-to-End Workflows](#12-end-to-end-workflows)
13. [Safety Model](#13-safety-model)
14. [File Index](#14-file-index)

---

## 1. What This Project Is

**Drift Triage Co-Pilot** is a self-healing MLOps stack for a bank marketing classification model. When deployed, it:

1. **Serves predictions** via a REST API (the sklearn model lives in memory)
2. **Accumulates prediction history** in-memory as it receives traffic
3. **Detects distribution drift** by computing PSI (Population Stability Index) and chi-squared on the accumulated history
4. **Auto-triages severity** and sends a webhook to an agent when drift changes
5. **Runs a deterministic investigation** (with optional LLM summarization) that recommends an action
6. **Dispatches safe actions** (replay test, retrain) to a Redis job queue
7. **Gates production-impacting actions** (promote, rollback) behind Human-in-the-Loop approval tracked in Postgres
8. **Provides a dashboard** for operators to monitor health, approve/reject actions, view model metrics, and trigger rollbacks

### The core problem it solves

Production ML models silently degrade as real-world data distributions shift. Without monitoring, operators learn about failures weeks later through business metrics, not ML metrics. Even with monitoring, without automated **and governable** corrective action, the gap between detection and resolution is manual and error-prone.

### What makes this different from a simpler approach

| Naive approach | This project |
|----------------|--------------|
| One monolith does everything | 8 independent microservices, each with one responsibility |
| Alert, then wait for human | Automatically triages, dispatches safe corrections immediately |
| LLM decides what to do | LLM only writes summaries; action routing is hardcoded |
| Auto-deploy retrained models to Production | Retrain creates a **candidate** only; promotion is HIL-gated |
| No audit trail | Every promotion/rollback written to Postgres before MLflow mutation |
| "v2 beats v1? Deploy" | Promotion checklist validates artifacts, schema, recall, and candidate alias |

---

## 2. Architecture: Why 8 Microservices

### The microservices

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Dashboard  │     │   Agent     │     │   Worker    │
│  :8501      │     │   :8001     │     │  (no port)  │
│  Operator   │     │  LangGraph  │     │  Redis      │
│  UI         │     │  Triage +   │     │  Consumer   │
│             │     │  HIL Routes │     │             │
└──────┬──────┘     └─────┬───────┘     └──────┬──────┘
       │                  │                    │
       │     ┌────────────┴──────────┐         │
       └─────┤        Platform        ├─────────┘
             │        :8000           │
             │  Predict, Drift,       │
             │  Registry, Promote,    │
             │  Rollback              │
             └───────┬────────────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────┴────┐ ┌────┴────┐ ┌───┴──────┐
    │ MLflow  │ │Postgres │ │  Redis   │
    │  :5000  │ │ :5432   │ │  :6379   │
    │Registry │ │HIL +    │ │Queue +   │
    │+ Artif. │ │Audit    │ │DLQ       │
    └─────────┘ └─────────┘ └──────────┘
```

### Why not a monolith?

**Separation of failure domains.** Each service fails independently. If MLflow goes down, the Platform still serves predictions from its in-memory model. If Redis goes down, the Agent still responds to webhooks (it will log dispatch errors but not crash). If the Worker crashes, pending jobs stay in the Redis queue (persisted to disk via Redis volume).

**Different resource profiles.** The Worker needs heavy ML dependencies (scikit-learn, pandas, numpy) to train models. The Dashboard only needs streamlit and requests. If they were one service, a training job would block the UI from loading.

**Independent scaling.** You could run 5 Workers to process a backlog, while keeping one Platform instance. You could run the Dashboard locally and the Platform in the cloud.

**Separation of ML dependency chains.** The Agent depends on langgraph/langchain. The Platform depends on sklearn/mlflow. The Worker depends on both. If these were one service, you'd have a dependency conflict nightmare.

### Why Docker Compose instead of Kubernetes?

This is a **demo/development project** with 8 services on a single machine. Docker Compose provides:
- Declarative `depends_on` with `condition: service_healthy` — services start in order
- Shared bridge network (`drift-net`) — DNS-based service discovery (`http://agent:8001`)
- Volume mounts for data and code sharing
- `.env` file for configuration
- Simple `docker compose up -d` startup

Kubernetes would be overkill for a single-node deployment. The architecture is **Kubernetes-ready** (stateless services, external state stores) but doesn't need it.

---

## 3. Technology Choices

### FastAPI — Why not Flask or Django?

**Performance.** FastAPI is built on Starlette and runs async natively. Every endpoint in this project is `async def` — meaning the server can handle concurrent requests while waiting for I/O (Postgres queries, Redis operations, HTTP calls to other services).

**Automatic validation.** FastAPI + Pydantic gives automatic request validation, OpenAPI schema generation (used by the dashboard health checks), and structured error responses (no stack traces in production).

**Dependency injection.** FastAPI's `Depends()` system lets us inject `app.state` singletons (model, threshold, HTTP client) into route handlers without global state or singleton classes.

```python
# Zero globals. Everything flows through Depends.
async def predict(body: PredictRequest, model=Depends(get_model), threshold=Depends(get_threshold)):
```

This is testable — you can override dependencies in tests without mocking modules.

### LangGraph — Why not a simple if-else chain?

The agent uses LangGraph's `StateGraph` with a **supervisor topology** (not a linear chain). This means:

```
START → supervisor → triage → supervisor → action → supervisor → execute_action → supervisor → comms → END
```

Every node returns to the supervisor, which routes to the next node based on state. This is more complex than:

```python
if severity == "critical": dispatch_retrain()
elif severity == "moderate": dispatch_replay()
```

But:

**Extensibility.** Adding a new node (e.g., "notify Slack") means adding one function and one routing condition. In an if-else chain, you'd need to restructure the flow.

**Checkpointing.** The repo now persists investigations and checkpoint state in Postgres and reuses the same investigation by `drift_event_id`. The optional LangGraph Postgres saver helper still exists, but the proven live recovery path is the repo-owned Postgres persistence flow.

**Observability.** LangSmith integration traces every node execution, making the decision path visible.

**The supervisor is deterministic.** The routing logic in `run_supervisor.py` is a pure function with no LLM calls:

```python
if state.get("comms_summary"): return "end"
if status in ("resolved", "failed"): return "comms"
if not state.get("triage_summary"): return "triage"
if not state.get("recommended_action"): return "action"
# etc.
```

### Redis — Why a queue instead of calling the worker directly?

**Decoupling.** The Agent doesn't need to know if the Worker is running. It pushes a job to Redis and returns immediately. The Worker picks it up when ready. If the Worker is down, jobs accumulate in the queue.

**Idempotency at the queue level.** Both the Agent (SADD to idempotency set) and Worker (SETNX with TTL) check for duplicates. This prevents double-processing if a webhook is received twice.

**Dead-letter queue.** Failed jobs go to `DLQ:drift-triage-jobs` for debugging, not lost.

**Persistence.** Redis volumes persist the queue to disk. If Redis restarts, jobs are not lost.

### Postgres — Why not just use MLflow for everything?

MLflow stores model metadata (versions, metrics, aliases). It does **not** store:
- Approval history (who approved what, when)
- Investigation records (what drift was detected, what action was recommended)
- Idempotency keys for HIL approvals
- `previous_version` for rollback audit trail

Postgres is the **durable audit store**. MLflow is the **model artifact/metadata store**. They serve different purposes.

Specifically: MLflow aliases are **overwritten** — when you change `Production` from v3 to v19, the old value is gone forever. Postgres's `promotion_audit` table preserves the full promotion history including what version Production pointed to before each change.

### MLflow — Why not just pickle the model to a file?

MLflow provides:
- **Versioned model registry**: Multiple model versions coexist with aliases (`Production`, `candidate`)
- **Run tracking**: Every training run logs metrics, parameters, and artifacts automatically
- **Artifact storage**: Model files, schema.json, model_card.json stored per version
- **REST API**: Other services can query "what's the current Production model?" via HTTP

Without MLflow, every service would need its own file-based registry and tracking. MLflow is the single source of truth for model metadata.

### Streamlit — Why not React/Vue/Svelte?

Streamlit is **Python-native**. The dashboard developer doesn't need JavaScript, CSS, or a build system. The entire UI is 364 lines of Python.

For an internal operations dashboard (not a customer-facing product), this is the right trade-off: developer speed over pixel-perfect design. Streamlit's `@st.cache_data` gives automatic API response caching without writing a cache layer.

### scikit-learn (HistGradientBoostingClassifier) — Why not XGBoost/LightGBM?

- **No binary dependencies.** `HistGradientBoostingClassifier` is pure Python (in scikit-learn 1.0+). XGBoost and LightGBM require C++ libraries that complicate Docker builds.
- **Native categorical support.** Handles categorical features without manual encoding (used in the preprocessor with OneHotEncoder).
- **Fast enough.** For 41K rows, it trains in seconds. Gradient-boosted trees handle class imbalance well (only 11% positives).
- **`class_weight="balanced"`** automatically adjusts for the imbalanced target.

### asyncpg — Why not psycopg2 or SQLAlchemy?

- **Native async.** The platform needs to query Postgres from `async def` endpoints without blocking the event loop.
- **Direct SQL.** The queries are simple INSERT/SELECT — an ORM would be overhead. Raw SQL with parameterized queries is simpler and more explicit.
- **Connection pooling not needed.** Each operation opens and closes a connection (read queries are infrequent, write queries are critical path). Pooling would add complexity without benefit.

### pydantic-settings — Why not os.getenv() everywhere?

```python
# BAD: scattered configuration
mlflow_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")
postgres_dsn = os.getenv("PG_DSN", "...")

# GOOD: centralized, validated
class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://mlflow:5000"
    extra = "forbid"  # rejects unknown env vars
```

Centralized config means:
- One place to look for all configuration
- Validation at startup (typos caught immediately)
- `extra="forbid"` rejects misnamed env vars (safety)
- Testing: override any setting without touching env

---

## 4. Platform Service (port 8000)

The platform is the **model serving layer** and **registry gatekeeper**. It has 4 responsibilities: prediction, drift detection, registry management, and queue status.

### File: `platform/app/main.py`

**`lifespan(app)`** — FastAPI lifespan context manager:

```python
app.state.model = load_model(settings.resolved_model_path())  # sklearn Pipeline
app.state.threshold = settings.threshold                        # operating threshold (0.3493)
app.state.http_client = httpx.AsyncClient(timeout=30.0)        # shared connection pool
app.state.drift_accumulator: list[dict] = []                   # in-memory prediction history
app.state.last_severity: str = "stable"                        # webhook suppression tracker
```

**Why `app.state` instead of a global variable?** FastAPI's `app.state` is the documented way to store application state. Global variables are not thread-safe in async contexts. `app.state` is accessible via `Depends()` in every route:

```python
async def predict(request: Request):
    state = request.app.state
    state.drift_accumulator.append(record)  # safe because a single async server is single-threaded
```

**Why a shared `httpx.AsyncClient`?** Creating a new HTTP client per request is wasteful (TCP handshake overhead). A single client with connection pooling is reused for all webhook emissions and inter-service calls. The client is closed in `lifespan` shutdown.

### File: `platform/app/dependencies.py`

**Pattern: FastAPI Depends()**

```python
def get_settings(request: Request) -> Settings:       # Read settings from app.state
async def get_model(request: Request):                 # Read model from app.state
async def get_threshold(request: Request) -> float:    # Read threshold from app.state
async def get_http_client(request: Request):           # Yield shared HTTP client
```

**Why these are separate functions instead of one `get_dependencies()`?** Each route should only depend on what it needs. The drift endpoint needs `settings` and `http_client` but not `model`. The predict endpoint needs `model` and `threshold` but not `http_client`. Separate dependencies = clearer contracts = easier to test.

**Why `get_http_client` is a generator (yield)?** FastAPI's async generator pattern ensures proper cleanup. If we needed to close the client per-request, the `finally` block would handle it.

### File: `platform/app/config/settings.py`

**`extra="forbid"`** — This is a safety mechanism. If someone sets `MLFLOW_URI` instead of `MLFLOW_TRACKING_URI`, the app crashes at startup with a clear error message rather than silently using the wrong default.

**`resolved_model_path()` and `resolved_dataset_path()`** — Path resolution handles both Docker (where files are at `/app/data/`) and local development (where they're relative to the platform directory). The fallback dataset path handles the case where the dataset was moved.

**`drift_severity_moderate: 0.10` and `drift_severity_critical: 0.25`** — These are the PSI thresholds. PSI > 0.10 is "moderate", PSI > 0.25 is "critical". These are industry-standard values (0.10 = small shift, 0.25 = large shift). They can be overridden via `.env`.

### File: `platform/app/routers/predict.py`

**`FEATURE_COLUMNS`** — Ordered list of 19 feature names + 1 derived feature (`pdays_never_contacted`). This ordering is critical because scikit-learn's Pipeline expects columns in this exact order. If the order changes between training and prediction, the model silently produces garbage predictions.

**`predict()` function:**

```python
row = pd.DataFrame([{...}])        # Step 1: Build 1-row DataFrame
row["pdays_never_contacted"] = ... # Step 2: Add derived feature
proba = model.predict_proba(row)   # Step 3: Get probability [P(no), P(yes)]
prediction = int(proba >= 0.3493)  # Step 4: Apply operating threshold
state.drift_accumulator.append({   # Step 5: Store for drift computation
    "age": ..., "euribor3m": ..., "proba": proba
})
```

**Why `predict_proba` + manual threshold instead of `model.predict()`?** `predict()` uses the default 0.5 threshold, which is wrong for imbalanced data (only 11% positive). The operating threshold (0.3493) was tuned via 5-fold CV to maximize recall while maintaining precision. We need the raw probability to apply the custom threshold.

**Why append to `drift_accumulator` on every prediction?** This is the data source for drift detection. By accumulating real prediction inputs and outputs, the drift report compares "what we saw before" (reference window) vs "what we're seeing now" (current window). Without this accumulator, drift detection would have no data.

**Why keep the accumulator in memory instead of a database?** For a demo, in-memory is sufficient. The accumulator is a rolling window of at most 500 records (< 100KB). For production, this would move to a time-series database or a Redis stream.

### File: `platform/app/routers/drift.py`

**`compute_psi(reference, current, bins=10)`**

PSI (Population Stability Index) measures how much two distributions differ:

```
PSI = Σ (P_ref - P_cur) × ln(P_ref / P_cur)
```

1. Histogram both distributions into 10 bins
2. Add Laplace smoothing (+1e-10) to avoid division by zero
3. Compute PSI formula

**Why 10 bins?** Industry standard for PSI computation. More bins = more sensitive to small shifts but noisier. Fewer bins = smoother but less sensitive. 10 is the sweet spot.

**Why Laplace smoothing?** If a bin has zero count in either distribution, `ln(0)` = -∞. Smoothing avoids this.

**`compute_chi2(reference, current)`**

Chi-squared tests whether two categorical distributions are independent:
1. Build a contingency table (reference counts × current counts)
2. Compute expected counts based on row/column totals
3. χ² = Σ (observed - expected)² / expected

**Why chi-squared instead of PSI for categoricals?** PSI works on continuous/numeric features. Chi-squared is the standard test for categorical features. The chi-squared statistic measures association strength — higher values mean more drift.

**`drift_report_to_alert(report, settings)`**

Converts the internal `DriftReport` to the external `DriftAlert` webhook shape. This is a **contract boundary** — the platform's internal representation doesn't need to match the agent's expected format. The conversion function is the adapter.

**`emit_webhook(report, client, settings)`**

POSTs the DriftAlert to the agent. Uses the shared `httpx.AsyncClient`. Returns `(success, error_message, response_body)` tuple. This is called **only when severity has changed** from the last recorded severity.

**`get_report()`**

The main drift report endpoint:

```python
accumulator = app_state.drift_accumulator
if len(accumulator) < 50:
    return insufficient data  # Need minimum sample size

# Split into reference (first half) and current (second half)
half = len(accumulator) // 2
ref_data = accumulator[:half]
cur_data = accumulator[half:]

# Compute PSI for each numeric feature
for feat in NUMERIC_FEATURES:
    psi_scores[feat] = compute_psi(ref_vals, cur_vals)

# Compute chi2 for each categorical feature
for feat in CATEGORICAL_FEATURES:
    chi2_scores[feat] = compute_chi2(ref_vals, cur_vals)

# Compute output drift (PSI on prediction probabilities)
output_drift = compute_psi(ref_probas, cur_probas)

# Classify severity
worst = max(max_psi, max_chi2, output_drift)
if worst >= 0.25: severity = "critical"
elif worst >= 0.10: severity = "moderate"
else: severity = "stable"

# Only emit webhook if severity changed
if severity != last_severity:
    emit_webhook(report, client, settings)
    app_state.last_severity = severity

# Clear accumulator for next window
app_state.drift_accumulator = []
```

**Why clear the accumulator after each report?** Each drift report represents a complete analysis of one window of predictions. Clearing the accumulator starts a fresh window for the next analysis. This prevents old data from diluting new drift signals.

**Why 50 as minimum sample size?** PSI is unreliable with small samples (the bins become too sparse). 50 is a conservative minimum. In production, this would be higher (200+).

**Why suppress webhooks when severity is unchanged?** Without suppression, every drift report would trigger a webhook, even if nothing changed. This would flood the agent with redundant alerts. The `last_severity` tracker is a simple deduplication mechanism.

### File: `platform/app/routers/registry.py`

This is the most complex router. It manages the model registry lifecycle.

**`_pg_dsn(settings)`**

Converts the asyncpg DSN format (`postgresql+asyncpg://`) to standard format (`postgresql://`). asyncpg's default DSN includes the `+asyncpg` scheme marker; removing it makes the DSN work with the standard `asyncpg.connect()` call.

**`_fetch_production_metrics(settings)`**

Queries MLflow for the current Production model's run metrics:
```python
prod = client.get_model_version_by_alias("bank_marketing_pipeline", "Production")
run = client.get_run(prod.run_id)
metrics = run.data.metrics  # {test_recall: 0.78, test_f1: 0.35, ...}
```

**Why get metrics from the run instead of the model version?** MLflow stores metrics at the **run** level, not the model version level. A model version links to a run. To get the metrics, you fetch the run linked to the model version.

**`_capture_current_production(settings)`**

Before any promotion or rollback, captures the current Production version:
```python
prod = client.get_model_version_by_alias("bank_marketing_pipeline", "Production")
return str(prod.version)
```

**Why capture before mutation?** After we change the alias, the old value is lost. We need to know what version we were on before the change for the audit trail's `previous_version` column.

**`_reload_model(request, settings)`**

After a promotion or rollback, reloads the model from MLflow into memory:
```python
model = mlflow.sklearn.load_model(f"models:/bank_marketing_pipeline@Production")
request.app.state.model = model
request.app.state.threshold = run_metrics["operating_threshold"]
```

**Why reload instead of restarting?** Restarting the container takes seconds and causes a brief outage. Hot-reloading takes milliseconds with zero downtime. The platform continues serving predictions from the old model until the reload completes.

**Why `mlflow.sklearn.load_model()` instead of `joblib.load(model.joblib)`?** After a rollback, the local `model.joblib` file can still contain a newer retrained model rather than the rolled-back Production version. MLflow's artifact store has every registered version. Loading from MLflow ensures we always load the version pointed to by the current alias.

**`_previous_production_version(settings)`**

Queries the `promotion_audit` table for the most recent audit row's `previous_version`:
```sql
SELECT previous_version FROM promotion_audit
WHERE previous_version IS NOT NULL
ORDER BY timestamp DESC LIMIT 1
```

**Why query `previous_version` directly instead of second-most-recent row?** Row order is fragile — if a promotion fails after audit write but before alias change, the audit row exists but doesn't represent a real alias change. `previous_version` is explicitly set at mutation time, so it's always correct.

**`GET /status`**

Returns the full registry state:
```json
{
    "registered_model_name": "bank_marketing_pipeline",
    "production_version": "26",
    "production_metrics": {"test_recall": 0.781, "test_f1": 0.355, ...},
    "previous_production_version": "1",
    "candidate_version": "20",
    "last_promotion": "2026-05-07T...",
    "status": "ok"
}
```

**Why include `production_metrics`?** The dashboard needs to show model health to operators (recall, F1, AUC). Without this, operators would need to open MLflow separately to check metrics.

**Why include `previous_production_version`?** The dashboard needs to know "if something goes wrong, what version do we roll back to?" This field directly answers that question.

**`POST /promote`**

```python
# 1. Validate approved_by is not empty
# 2. Run promotion checklist (artifacts, metrics, candidate alias)
# 3. Capture current production version (for audit)
# 4. Write audit to Postgres (DURABLE — before MLflow mutation)
# 5. Set Production alias in MLflow
# 6. Hot-reload model from MLflow into memory
# 7. Log to structlog
```

**Why write audit BEFORE MLflow mutation?** If the audit write fails (Postgres down), we abort before changing Production. If the MLflow mutation fails, we have a partial audit — but the retry will be idempotent (same model_uri). The alternative (MLflow first, then audit) risks changing Production without recording it.

**`POST /rollback`**

```python
# 1. Validate approval in Postgres:
#    - approval_id exists
#    - status is "approved"
#    - requested_action is "rollback"
#    - target_model_version matches (if present in approval)
# 2. Validate target version exists in MLflow
# 3. Capture current production version
# 4. Write rollback audit to Postgres
# 5. Set Production alias to target version
# 6. Hot-reload model
# 7. Log to structlog
```

**Why validate the approval against Postgres?** Just because a rollback request arrives with an `approval_id` doesn't mean it's valid. The approval could be:
- Non-existent (404)
- Still pending (409 — someone tried to rollback before approval)
- For a different action like promote_candidate (409)
- For a different target version (409 — cannot roll back to one version with an approval issued for another)

Every case is an explicit error, not a silent failure.

**`_validate_rollback_approval(settings, body)`**

```python
row = await conn.fetchrow("""
    SELECT approval_id, investigation_id, requested_action,
           target_model_version, status, approved_by
    FROM hil_approvals WHERE approval_id = $1
""", body.approval_id)

if row is None: raise 404
if row["status"] != "approved": raise 409
if row["requested_action"] != "rollback": raise 409
if row["target_model_version"] != body.target_version: raise 409
```

**Why validate `target_model_version` match?** An approval for one rollback target should not allow rolling back to some other arbitrary version. This prevents a valid approval ID from being reused for the wrong Production mutation.

**`GET /history`**

Returns all promotion audit records ordered by timestamp. The dashboard uses this to populate the Promotion History expander.

**`_write_audit_to_postgres(settings, body, previous_version)`**

```sql
INSERT INTO promotion_audit
(model_uri, investigation_id, approved_by, from_alias, to_alias, previous_version)
VALUES ($1, $2, $3, $4, $5, $6)
```

**Why parameterized queries ($1, $2, ...) instead of string formatting?** SQL injection prevention. Parameterized queries separate SQL structure from data values. Even though the values come from our own code (not user input), it's a security habit.

**`_write_rollback_audit(settings, body, approval, previous_version)`**

Similar but uses the approval's `investigation_id` instead of a placeholder. The `from_alias` and `to_alias` are both "Production" for rollbacks (we're changing Production from X to Y).

### File: `platform/app/routers/queue.py`

A lightweight endpoint that returns Redis queue status:
```python
{"queue_length": 3, "dlq_length": 0, "redis_connected": true}
```

**Why a separate `/queue/status` endpoint?** The dashboard needs to show queue health without accessing Redis directly. This endpoint is the contract between the dashboard and the queue.

### File: `platform/app/services/run_training.py`

This is the full ML training pipeline, importable by both the platform and the worker. 279 lines.

**`load_and_clean(csv_path)`**

```python
df = pd.read_csv(csv_path, sep=";")
y = (df["y"] == "yes").astype(int).values
df.drop(columns=["y", "duration"], inplace=True)
df["pdays_never_contacted"] = (df["pdays"] == 999).astype(int)
```

**Why drop `duration`?** Duration is a target leak — it measures the call duration and is only known after the call completes. Including it would make the model appear artificially good but fail in production where duration is unknown.

**Why `pdays_never_contacted`?** `pdays=999` means "never contacted before." This is a sentinel value that should not be treated as a numeric quantity. A binary flag captures the semantics correctly.

**`find_threshold(y_true, y_proba, min_recall=0.75)`**

```python
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
valid = recall[:-1] >= min_recall
return thresholds[valid].max()
```

**Why precision_recall_curve?** The bank marketing use case cares about recall (finding as many "yes" customers as possible) while maintaining acceptable precision. The threshold is the highest value where recall >= 0.75.

**5-fold CV for threshold:**

```python
for train_idx, val_idx in skf.split(X_trainval, y_trainval):
    # Fit on train fold
    pp = ColumnTransformer(...)
    est = HistGradientBoostingClassifier(...)
    est.fit(X_tr_t, y_tr)
    # Find threshold on val fold
    t = find_threshold(y_v, y_v_proba, 0.75)
    thresholds.append(t)
operating_threshold = np.mean(thresholds)
```

**Why cross-validate the threshold?** The threshold should generalize, not overfit to one split. CV averages the threshold across 5 folds, giving a more robust value.

**MLflow logging:**

```python
mlflow.log_dict(schema, "schema.json")
mlflow.log_dict(model_card, "model_card.json")
mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", registered_model_name="...")
client.set_registered_model_alias(name="...", alias="candidate", version=...)
```

**Why log schema.json?** The promotion checklist validates that the model artifact includes a schema. This ensures the model is documented and can be audited — someone can look at the schema and know exactly what features the model expects.

**Why log model_card.json with SHA256?** The SHA256 hash of the dataset proves that the model was trained on a specific dataset. If someone claims "the model's bad because the data changed," you can compare hashes. The environment fingerprint (Python version, sklearn version, etc.) documents the training environment for reproducibility.

**Why `candidate` alias, never `Production`?** This is the core safety invariant. Training never touches Production. It creates a new version with the `candidate` alias only. Production is changed only through the HIL-gated promotion flow.

### File: `platform/app/services/validate_promotion.py`

**`assert_promotion_checklist(model_uri)`**

5 checks that must pass before promotion:

1. **Model exists** in registry at `@candidate`
2. **schema.json artifact** present and non-empty
3. **model_card.json** with `dataset.sha256` and `environment` fingerprint
4. **test_recall >= 0.75** (minimum quality bar)
5. **Candidate alias** present (model went through candidate stage)

**Why ignore the `model_uri` parameter and hardcode `@candidate`?** The promotion gate always promotes the current candidate. This prevents someone from promoting an arbitrary version by passing a different model_uri. The API accepts `model_uri` for forward compatibility but the validation always checks the candidate.

**Why check `dataset.sha256` in model_card?** This proves the model card was generated by our training pipeline (which includes the hash). A hand-edited model card would fail this check.

### File: `platform/app/services/compute_drift.py`

Standalone drift computation module. Uses `scipy.stats.chi2_contingency` for proper chi-squared test. This is an alternative implementation to the inline versions in `drift.py` — both exist and produce equivalent results.

---

## 5. Agent Service (port 8001)

The agent is the **decision-making layer**. It receives drift webhooks, runs a LangGraph investigation, dispatches actions, and manages HIL approvals.

### Design philosophy: LLM is advisory, not executive

The LLM (Azure OpenAI Kimi-K2.6-1) is used **only** for text summarization — writing triage summaries and communication messages. It has **zero influence** on what action gets dispatched. The action routing is a hardcoded mapping:

```python
# run_action.py — pure deterministic mapping
if severity == "stable":    action = "none"
if severity == "moderate":  action = "replay_test"
if severity == "critical":  action = "retrain"
```

The LLM can write "Critical drift detected on euribor3m and cons_price_idx. The model's predictions have shifted significantly from the reference distribution" instead of the deterministic "Critical drift detected. Retraining candidate should be considered." But it cannot change `retrain` to `promote_to_production`. That safety boundary is enforced in `run_execute_action.py`:

```python
PRODUCTION_ACTIONS = {"rollback", "promote_candidate"}
# retrain is NOT in this set → dispatched to Redis queue
# rollback/promote_candidate ARE → create HIL approval, NEVER executed directly
```

### File: `agent/app/graph/state.py`

**`AgentState`** — a TypedDict that threads through all LangGraph nodes:

```python
class AgentState(TypedDict):
    investigation_id: str          # UUID for this investigation
    drift_event_id: str            # Correlated drift event
    drift_alert: DriftAlert        # Incoming webhook payload
    severity: Severity             # stable/moderate/critical
    triage_summary: str | None     # Filled by triage node
    recommended_action: ... | None # Filled by action node
    comms_summary: str | None      # Filled by comms node
    job_id: str | None             # Redis job ID if queued
    queued: bool | None            # Whether job was enqueued
    dispatch_error: str | None     # Error if dispatch failed
    approval_id: str | None        # HIL approval ID if created
    requires_approval: bool        # Whether this needs human review
    status: InvestigationStatus    # Lifecycle: open → queued/approved → resolved
```

**Why TypedDict instead of Pydantic BaseModel?** LangGraph's state management works best with dict-like objects. TypedDict gives type safety at development time without runtime overhead.

### File: `agent/app/graph/build_graph.py`

**`build_agent_graph()`**

Constructs the StateGraph with 5 nodes and conditional edges:

```python
graph = StateGraph(AgentState)
graph.add_node("triage", run_triage)
graph.add_node("action", run_action)
graph.add_node("execute_action", run_execute_action)
graph.add_node("comms", run_comms)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", supervisor_node, {
    "triage": "triage", "action": "action",
    "execute_action": "execute_action", "comms": "comms", "end": END
})
# Every node returns to supervisor:
for node in ["triage", "action", "execute_action", "comms"]:
    graph.add_edge(node, "supervisor")
```

**Why every node returns to supervisor?** This is a **supervisor topology**, not a linear chain. The supervisor re-evaluates state after each node and decides the next step. This allows:
- Graceful error handling (if execute_action fails, supervisor routes to comms)
- Conditional skipping (if severity is stable, skip execute_action)
- Future extensibility (add a "notify_slack" node without changing existing nodes)

**`supervisor_node(state)`**

```python
def supervisor_node(state):
    if state.get("comms_summary"): return {"next_node": "end"}
    if state["status"] in ("resolved", "failed"): return {"next_node": "comms"}
    if not state.get("triage_summary"): return {"next_node": "triage"}
    if not state.get("recommended_action"): return {"next_node": "action"}
    if state.get("recommended_action") == "none" and not state.get("queued"):
        return {"next_node": "comms"}
    # ... etc
```

**Why is the supervisor deterministic?** The supervisor is a routing function, not a decision-making function. It doesn't decide WHAT to do — it decides WHERE to go next based on what's already been done. Making it deterministic ensures predictable execution order.

**`run_investigation(body)`**

The entry point for webhook processing:
```python
graph = build_agent_graph()
initial = _initial_state(body)
config = {"configurable": {"thread_id": initial["investigation_id"]}}
result = await graph.ainvoke(initial, config=config)
return result
```

**Why reuse `investigation_id` per `drift_event_id`?** The repo reuses the same persisted investigation state when the same drift event is delivered again. That keeps retries idempotent and lets the graph continue from saved Postgres state instead of creating a second investigation for the same event.

### File: `agent/app/graph/run_triage.py`

**Pattern: LLM with deterministic fallback**

```python
if severity == "stable" or provider == "mock":
    # Skip LLM entirely — use deterministic summary
    llm_result = fallback
else:
    try:
        llm_result = complete_json(
            system_prompt="Summarize the drift alert... Do not recommend production changes.",
            user_payload={drift details},
            fallback=fallback,
            output_model=TriageOutput,
        )
    except RuntimeError:
        llm_result = fallback  # LLM failed, use deterministic fallback
```

**Why skip LLM for "stable"?** Stable means "nothing is wrong." Calling an LLM for a "no issues" report wastes API calls and introduces latency. The deterministic text is sufficient.

**Why system prompt says "Do not recommend production changes"?** This is a guardrail. Even though the LLM's output doesn't determine the action (that's `run_action.py`'s job), the summary text could confuse operators if it suggested production changes. The system prompt prevents this.

**Why a fallback on every path?** LLMs are unreliable. Network failures, rate limits, API errors — all possible. The deterministic fallback ensures the graph never fails because the LLM is unavailable. The system degrades gracefully: less elegant summaries, but identical behavior.

### File: `agent/app/graph/run_action.py`

```python
def run_action(state: AgentState) -> AgentState:
    severity = state["severity"]
    if severity == "stable":
        recommended_action = "none"; status = "resolved"
    elif severity == "moderate":
        recommended_action = "replay_test"; status = "open"
    else:
        recommended_action = "retrain"; status = "open"
```

**Why is this a separate node from triage?** Separation of concerns. Triage PRODUCES a human-readable summary. Action DECIDES what to do. These are different responsibilities. Keeping them separate makes each function testable in isolation.

**Why is this pure deterministic (no LLM, no API calls)?** Action routing is a safety-critical decision. A hallucinating LLM could recommend "promote to Production" on every drift alert. The hardcoded mapping cannot hallucinate.

**Why does "stable" resolve immediately?** Stable = no drift = nothing to do. Marking it as "resolved" tells the supervisor to skip execute_action and go straight to comms.

### File: `agent/app/graph/run_execute_action.py`

```python
if action == "none":
    updated["status"] = "resolved"

elif action == "replay_test":
    dispatch_result = await dispatch_replay_test(...)
    updated["status"] = "queued" if queued else "open"

elif action == "retrain":
    dispatch_result = await dispatch_retrain(...)
    updated["status"] = "queued" if queued else "open"

elif action in PRODUCTION_ACTIONS:  # {"rollback", "promote_candidate"}
    approval = await create_pending_approval(...)
    updated["requires_approval"] = True
    updated["status"] = "waiting_for_approval"
```

**Why are `rollback` and `promote_candidate` special-cased?** These actions change Production. The agent MUST NOT execute them directly — even though the deterministic action policy never maps severity to these actions, a future code change or a manually-created investigation could. The `PRODUCTION_ACTIONS` set is a defense-in-depth measure: if code somehow generates a `rollback` recommendation, it will be gated behind HIL approval, not executed.

**Why catch all exceptions with `_safe_error_message`?** We never want stack traces in the API response. `_safe_error_message()` returns the error message without the traceback, which is safe to show to operators.

```python
except Exception as exc:
    updated["status"] = "failed"
    updated["dispatch_error"] = _safe_error_message(exc)
    updated["requires_approval"] = action in PRODUCTION_ACTIONS
    return updated
```

### File: `agent/app/graph/run_comms.py`

Builds a final communication summary. Similar LLM-with-fallback pattern as triage.

**Why a separate comms node?** Comms is the final step after all actions are taken. Its summary includes the investigation outcome (queued/failed/needs_approval). It's separate from triage (which summarizes the drift) because the audience is different — triage for internal routing, comms for external reporting.

### File: `agent/app/routers/webhook.py`

Single endpoint: `POST /webhook/drift`

```python
async def receive_drift_alert(body: DriftAlert):
    state = await run_investigation(body)
    return {flattened state dict}
```

**Why return a flattened dict instead of the full state?** The webhook response is a contract with the platform. The platform needs specific fields (investigation_id, recommended_action, queued). Returning the full AgentState would expose internal fields and couple the platform to the agent's internals.

### File: `agent/app/routers/hil.py`

5 endpoints for the HIL approval lifecycle:

| Endpoint | Purpose |
|----------|---------|
| `GET /pending` | Dashboard polls this for the HIL Inbox |
| `GET /{id}` | Get a single approval |
| `POST /{id}/approve` | Human approves |
| `POST /{id}/reject` | Human rejects |
| `POST /notify-candidate` | Worker creates approval after retrain |

**`GET /pending`**

```python
approvals = await request_approval.list_pending_approvals(limit=50)
return {"approvals": [approval.model_dump(mode="json") for approval in approvals]}
```

**Why `model_dump(mode="json")`?** Pydantic's default serialization uses Python types. `mode="json"` converts datetimes to ISO strings and enums to their values, making the response ready for JSON consumption by the dashboard.

**`POST /{id}/approve`**

```python
updated = await request_approval.approve_action(
    approval_id=approval_id,
    approved_by=body.approved_by,
    reason=body.reason,
)
return ApprovalDecisionResponse(approval_id=..., status="approved", message="Approval approved")
```

**Why require `approved_by`?** Audit trail. Every approval records who approved it. In a regulated environment, this is required for compliance.

**`POST /notify-candidate`**

```python
approval = await request_approval.create_pending_approval(
    investigation_id=body.investigation_id,
    requested_action="promote_candidate",
    target_model_version=version,
    idempotency_key=f"worker:{investigation_id}:{version}",
)
```

**Why an idempotency key based on investigation_id + version?** If the worker retries the notification (due to a network error), the duplicate POST will return the existing approval instead of creating a second one. The `ON CONFLICT (idempotency_key) DO UPDATE SET idempotency_key = ...` pattern in Postgres handles this.

### File: `agent/app/services/request_approval.py`

**`create_pending_approval()`**

```sql
INSERT INTO hil_approvals (approval_id, investigation_id, ..., idempotency_key)
VALUES ($1, $2, ..., $7)
ON CONFLICT (idempotency_key) DO UPDATE SET idempotency_key = hil_approvals.idempotency_key
RETURNING *
```

**Why `ON CONFLICT DO UPDATE SET idempotency_key = idempotency_key`?** This is a no-op update that returns the existing row. It's a PostgreSQL pattern for "insert if not exists, return existing if exists" — a type of upsert that's idempotent.

**`validate_status_transition()`**

```python
if current_status != "pending":
    raise ValueError(f"Cannot transition approval from '{current_status}' to '{target_status}'.")
```

**Why enforce pending → approved/rejected only?** An approval is a one-time decision. Once approved or rejected, it cannot be changed. This prevents:
- Approving an already-approved approval (double-approval)
- Approving a rejected approval (the rejection was intentional)
- Rejecting an already-rejected approval (no-op)

### File: `agent/app/tools/queue_client.py`

**`enqueue_job(job_type, payload, idempotency_key)`**

```python
redis = await get_redis_client()
# Check idempotency
added = await redis.sadd("ops_job_idempotency_keys", idempotency_key)
if not added:
    return {"queued": False, "duplicate": True}  # Already dispatched

# Build job payload with retry config
job = build_job_payload(job_type, payload, idempotency_key)
job["max_attempts"] = 3
job["attempts"] = 0

# Push to queue
await redis.rpush("drift-triage-jobs", json.dumps(job))
return {"queued": True, "job_id": job["job_id"], ...}
```

**Why `SADD` instead of a simple key check?** `SADD` is atomic — it adds and checks in one operation. If two webhooks arrive simultaneously with the same idempotency key, both `SADD` calls happen atomically, and only one returns `added=1`. A separate `EXISTS` + `SADD` would have a race condition.

**Why `RPUSH` (right push) instead of `LPUSH`?** `RPUSH` adds to the right (tail) of the list. `BLPOP` removes from the left (head). Together, they form a FIFO queue — first in, first out. This is the standard pattern for work queues.

**Why `build_job_payload()` includes `max_attempts` and `attempts`?** The worker uses these fields for retry logic. Including them in the payload makes the retry behavior explicit and configurable per job.

### File: `agent/app/config/settings.py`

**`LLM_PROVIDER` defaults to `"mock"`**

**Why is "mock" the default?** Safety. If the `.env` file is missing or misconfigured, the agent runs in mock mode — deterministic summaries, no API calls, no cost. An incorrectly configured LLM provider in production mode could silently bill thousands of API calls.

**`extra="ignore"` (not "forbid")**

The agent is more permissive with env vars than the platform because it has many optional configuration keys (LangSmith, LangChain, various LLM providers). `extra="forbid"` would reject valid optional keys. `extra="ignore"` accepts unknown keys without crashing.

---

## 6. Worker Service (no port)

The worker is a **background process** — no HTTP server, no REST API. It connects to Redis and processes jobs in a loop.

### File: `worker/app/worker/consume_queue.py`

**`run_loop()`**

```python
async def run_loop():
    redis = aioredis.from_url(settings.redis_url)
    while True:
        result = await redis.blpop("drift-triage-jobs", timeout=5.0)
        if result is None:
            continue  # No jobs available, wait for next poll
        _, raw = result
        await process_job(redis, raw)
```

**Why `BLPOP` with timeout instead of `BRPOP`?** `BLPOP` pops from the left (head) — FIFO with `RPUSH`. The 5-second timeout means the loop checks for shutdown signals every 5 seconds (via `CancelledError`). Without a timeout, `BLPOP` would block indefinitely and couldn't be cancelled.

**`process_job(redis, raw)`**

```python
job = json.loads(raw)
action = normalize_action(job.get("action"))
idempotency_key = build_idempotency_key(action, investigation_id, idempotency_target(job))

# Idempotency check — SETNX with TTL
acquired = await redis.set(idempotency_key, "processing", nx=True, ex=3600)
if not acquired:
    return  # Already processed by another worker or previous attempt

# Execute with retries
for attempt in range(1, 4):
    try:
        await handler(job)
        return
    except Exception as exc:
        await asyncio.sleep(1 * (2 ** (attempt - 1)))  # 1s, 2s, 4s

# Final failure → dead-letter queue
await redis.rpush(DLQ_NAME, json.dumps(job))
```

**Why `SET nx ex` (SET if Not eXists with expiry)?** This is Redis's equivalent of a distributed lock. If another worker instance picks up the same job (due to duplicate dispatch or network partition), the SETNX fails and the duplicate is skipped. The 1-hour TTL prevents abandoned locks from accumulating forever.

**Why exponential backoff (1s → 2s → 4s)?** Immediate retries would hammer failing dependencies. Exponential backoff gives transient failures (network blips, DB restarts) time to recover. 3 retries is a standard choice — enough for transient failures, not so many that systemic failures waste resources.

**Why a dead-letter queue instead of dropping failed jobs?** Failed jobs contain the full payload — you can inspect, debug, and manually re-queue them. Dropping them loses information about what failed and why.

**`handle_retrain(job)`**

```python
model_uri = await loop.run_in_executor(None, run_training_pipeline, job.get("dataset_path"))
metrics = fetch from MLflow
await _notify_agent_candidate(investigation_id, drift_event_id, model_uri, metrics)
```

**Why `run_in_executor`?** `run_training_pipeline()` is a synchronous function that blocks for seconds (loading CSV, fitting sklearn). Running it in an executor thread prevents it from blocking the async event loop.

**Why fetch metrics after training?** The agent's `/notify-candidate` endpoint creates an approval with metrics. The dashboard needs these metrics to show the candidate model's quality (recall, F1, AUC).

**`handle_replay(job)`**

```python
model = joblib.load("data/model.joblib")
proba = model.predict_proba(df)[:, 1]
avg_score = float(np.mean(proba))
```

**Why replay the model?** A replay test verifies that the current Production model still works on a test dataset. It's a read-only check — no model changes, no artifact writes. It's safe to run automatically without approval.

**Why `joblib.load` instead of loading from MLflow?** The replay test is a quick validation, not a deployment. Loading from disk is faster than downloading from MLflow. The model.joblib is the current Production model (updated by retrain, but retrain saves to disk).

**`handle_rollback(job)`**

```python
if not approval_id: raise RuntimeError("Rollback refused: no approval_id")
if not target_version: raise RuntimeError("Rollback refused: no target_model_version")

resp = await client.post(f"{PLATFORM_BASE_URL}/registry/rollback", json={
    "target_version": target_version,
    "approval_id": approval_id,
    "approved_by": approved_by,
})
```

**Why two separate refusal checks?** Each check has a specific error message. "No approval_id" and "no target_version" are different problems requiring different fixes. Combined messages would be ambiguous.

**Why POST to `/registry/rollback` instead of calling MLflow directly?** The platform is the gatekeeper. The rollback endpoint validates the approval, writes the audit, sets the alias, and reloads the model — all as one atomic operation. Calling MLflow directly would bypass the audit and validation.

---

## 7. Dashboard Service (port 8501)

### File: `dashboard/app.py`

**`@st.cache_data(ttl=5)`**

```python
@st.cache_data(ttl=5)
def _get(url: str) -> dict:
    return requests.get(url, timeout=5).json()
```

**Why TTL=5?** 5 seconds is the right balance between freshness and API load. The dashboard typically refreshes every 5-10 seconds (when the user interacts), so a 5-second cache prevents redundant API calls while ensuring data is reasonably current.

**Why `st.cache_data.clear()` after approve/reject/rollback?** After a user action, the cached data is stale. Clearing the cache forces a fresh API fetch on the next render. Without this, the user might see the old approval list for up to 5 seconds after approving.

**Health bar (4 services):**

```python
for col, name, url in [(c1, "Platform", PLATFORM), (c2, "Agent", AGENT), ...]:
    ok = _health_ok(url)
    st.metric(f"{'✅' if ok else '❌'} {name}", "Connected" if ok else "Offline")
```

**Why show health on every page load?** The dashboard is an operations tool. If a service is down, the operator needs to know immediately before taking any action.

**Drift monitoring (3 demo buttons):**

```python
st.button("Normal (500)"):    _run_drift_demo("Normal", 500, 0, PREDICT_PAYLOAD)
st.button("Moderate Drift"):  _run_drift_demo("Moderate", 250, 250, MODERATE_SHIFT)
st.button("Critical Drift"):  _run_drift_demo("Critical", 100, 400, CRITICAL_SHIFT)
```

**Why 500 predictions?** PSI needs a reasonable sample size. 500 total predictions gives 250 reference and 250 current observations, which is sufficient for stable PSI computation.

**Why different split ratios per severity?** The severity is determined by the magnitude of the shift. For "Critical," we use 400 shifted predictions out of 500 (80% shifted) with large shifts on euribor3m and cons_price_idx. For "Moderate," we use 250 shifted (50% shifted) with smaller shifts. The ratio + magnitude together determine the PSI value.

**HIL Approval Inbox:**

```python
approvals_data = _get(f"{AGENT}/hil/pending")
for a in approvals:
    st.markdown(f"Action: {a['requested_action']}")
    st.metric("Candidate", reg.get("candidate_version"))
    st.metric("Recall", f"{recall:.3f}")
    # Approve/Reject buttons
```

**Why show candidate metrics in the approval card?** The operator needs data to make an informed decision. Showing recall/F1/AUC with a pass/fail indicator (recall >= 0.75) gives the operator confidence that the candidate model meets quality standards.

**Registry Status:**

```python
st.metric("Model", reg.get("registered_model_name"))
st.metric("Version", f"v{prod_ver}")
st.metric("Recall", f"{recall:.3f}")
st.metric("F1", f"{f1:.3f}")
st.metric("AUC", f"{auc:.3f}")
```

**Why show only 5 key metrics?** Operators don't need to see every training metric. Model name + version + the 3 key quality metrics (recall, F1, AUC) give a complete picture of "what model is running and how good is it?"

**Promotion History with Rollback:**

```python
with st.expander("Promotion History"):
    rec = records[0]  # Only the most recent
    st.caption(f"{rec['timestamp']} | {rec['from_alias']} → {rec['to_alias']} | v{ver} by {rec['approved_by']}")

    if prev:
        rollback_approval_id = st.text_input("Rollback approval ID")
        st.button("Rollback to v{prev}", disabled=not bool(approval_id))
```

**Why show only the most recent promotion?** The dashboard is for operations, not archaeology. The most recent promotion is what matters for rollback decisions — it tells you "what version was Production before the latest change?" If you need the full audit trail, pgAdmin has it.

**Why require an approval ID for the rollback button?** Rollback is a production-changing action. It MUST be HIL-gated. The operator needs a pre-approved rollback approval from the HIL system. This prevents accidental rollbacks — the approval ID serves as a confirmation that a second human approved the rollback.

---

## 8. MLflow Service (port 5000)

### File: `mlflow/Dockerfile`

```dockerfile
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000",
     "--serve-artifacts", "--backend-store-uri", "sqlite:///data/mlflow.db",
     "--default-artifact-root", "file:///data/mlruns", "--allowed-hosts", "*"]
```

**Why `--serve-artifacts`?** This enables MLflow's artifact proxy — the tracking server serves model artifacts via HTTP. Without this, other containers would need direct filesystem access to the artifact store. With it, they can download artifacts via REST API.

**Why SQLite instead of Postgres for the backend store?** MLflow's tracking data (runs, metrics, params) is operational data, not audit data. It doesn't need the durability of Postgres. SQLite is simpler, requires no additional database setup, and the data lives on the shared volume mount.

**Why `--allowed-hosts '*'`?** In a Docker network, services connect via internal hostnames (`mlflow:5000`). Without this, MLflow would reject requests from non-localhost sources. In production, this would be restricted to specific service names.

**Volume mount: `- ./platform/data:/data`**

This shares the platform's data directory with MLflow. The MLflow artifacts (models, schemas) are stored under `/data/mlruns/`, which is also accessible from the platform container (after we added the `/data` mount). This shared volume is how the platform reads MLflow artifacts.

---

## 9. Postgres + pgAdmin

### File: `postgres/init.sql`

**Three tables:**

| Table | Purpose | Key constraint |
|-------|---------|---------------|
| `investigations` | Drift investigation records | `investigation_id` PRIMARY KEY |
| `hil_approvals` | HIL approval lifecycle | `idempotency_key` UNIQUE, `approval_id` PRIMARY KEY |
| `promotion_audit` | Immutable promotion/rollback log | `id` SERIAL PRIMARY KEY |

**Why `idempotency_key` is UNIQUE on `hil_approvals`?** This is the database-level enforcement of idempotency. Even if the application layer fails to check for duplicates, the database will reject the duplicate INSERT. Two layers of defense.

**Why are `hil_approvals` indexed by status, investigation_id, and drift_event_id?** The most common query is `SELECT * FROM hil_approvals WHERE status = 'pending' ORDER BY created_at DESC` — the dashboard's HIL inbox. The status index makes this query fast. investigation_id index supports joining with investigations. drift_event_id index supports debugging by event.

**Why is `promotion_audit` indexed by timestamp DESC?** The `GET /history` endpoint queries `ORDER BY timestamp DESC LIMIT 50`. The descending index makes this a direct index scan.

**Why `previous_version TEXT NULL` in `promotion_audit`?** This column explicitly records what the Production version was before each promotion or rollback. It's THE column that makes the rollback button work — without it, the dashboard can't know what version to roll back to.

**pgAdmin on port 5050:**

A web-based database browser. The instructor can connect to `postgres:5432` and browse the 3 tables live during the demo. This demonstrates that the audit trail is real, not simulated.

---

## 10. Redis Service (port 6379)

**Why Redis for the job queue instead of RabbitMQ or Kafka?** Redis is already in the stack (used by the agent for idempotency). Adding RabbitMQ or Kafka would introduce a new technology that needs to be learned, configured, and maintained. For a demo with < 100 jobs at a time, Redis is perfectly sufficient.

**Redis data structures used:**

| Structure | Purpose |
|-----------|---------|
| List (`RPUSH`/`BLPOP`) | Job queue — `drift-triage-jobs` and `DLQ:drift-triage-jobs` |
| Set (`SADD`/`SISMEMBER`) | Idempotency tracking — `ops_job_idempotency_keys` |
| String (`SET nx ex`) | Per-job idempotency — `idempotency:{action}:{investigation_id}:{target}` |

**Why a Set for agent idempotency but a String for worker idempotency?** Different use cases:
- Agent: "Has this investigation already been dispatched?" → Set membership check
- Worker: "Is this specific job currently being processed?" → Distributed lock with TTL

The Set is permanent (until cleared), the String is temporary (1-hour TTL). This reflects their semantics — agent idempotency prevents duplicate dispatch forever, worker idempotency prevents concurrent processing for the duration of the job.

---

## 11. Cross-Cutting Design Decisions

### Idempotency Everywhere

| Layer | Mechanism | Scope |
|-------|-----------|-------|
| Webhook emission | `idempotency_key: {event_id}:{severity}` in DriftAlert | Platform ↔ Agent |
| Agent dispatch | `SADD` to Redis set with idempotency key | Agent → Redis |
| Worker processing | `SET nx ex` Redis string with idempotency key | Worker → Redis |
| HIL approval creation | `ON CONFLICT (idempotency_key)` in Postgres | Worker → Agent → Postgres |
| HIL status transitions | `validate_status_transition()` | Agent → Postgres |

**Why so many idempotency layers?** Distributed systems have no single "right" place for idempotency. Each service boundary is a potential failure point where a message could be delivered twice. Defense in depth.

### Defense in Depth: Production Safety

```
Layer 1: Deterministic action mapping (run_action.py)
         └─ severity → action is hardcoded. LLM cannot influence routing.

Layer 2: execute_action gating (run_execute_action.py)
         └─ PRODUCTION_ACTIONS set prevents direct execution of rollback/promote.
            Creates HIL approval instead.

Layer 3: HIL approval validation (hil.py + request_approval.py)
         └─ Status transition validation. Idempotent creation.

Layer 4: Rollback approval validation (registry.py _validate_rollback_approval)
         └─ 6 checks: exists, approved, rollback action, version match, etc.

Layer 5: Audit-first writes (registry.py promote/rollback)
         └─ Postgres audit written BEFORE MLflow alias change.
            If audit fails, alias change is aborted.

Layer 6: Model hot-reload (registry.py _reload_model)
         └─ After alias change, model reloaded from MLflow.
            If reload fails, alias change is NOT rolled back (audit remains).
            But the next successful reload will pick up the new model.
```

### Deterministic Fallbacks

Every LLM call has a non-LLM fallback:

| Function | LLM used | Fallback |
|----------|----------|----------|
| `run_triage.py` | Azure OpenAI (summary enrichment) | Hardcoded severity→summary map |
| `run_comms.py` | Azure OpenAI (communication text) | Hardcoded state→summary template |
| `run_action.py` | NONE (always deterministic) | N/A (safety-critical) |
| `run_execute_action.py` | NONE (always deterministic) | N/A (dispatches or creates approval) |
| `run_supervisor.py` | NONE (always deterministic) | N/A (pure state machine) |

**Why "mock" is the default LLM provider?** If `.env` is missing or misconfigured, the system runs in fully deterministic mode. This is safe — the system works correctly without an LLM, just with less elegant summaries.

### Lazy Imports for Heavy Dependencies

```python
# In request_approval.py
def _load_asyncpg():
    return importlib.import_module("asyncpg")
```

**Why lazy imports?** `asyncpg` and `redis` are heavy dependencies that:
- Require compilation or native libraries
- Are not needed for every code path
- Would cause import failures if not installed (e.g., in test environments)

Lazy imports mean the module can be imported (for type checking, IDE support) without the dependency being available at import time. Only when the function is called does the import happen.

### Centralized Configuration

```python
# BAD: scattered os.getenv()
mlflow_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")

# GOOD: pydantic-settings
class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://mlflow:5000"
    extra = "forbid"
```

**Why is this a recurring pattern?** In a microservice architecture with 5 services (platform, agent, worker, dashboard, mlflow), each with different config needs, centralized config classes prevent configuration drift. Each service imports ONE config object. All config values have defaults, types, and documentation in one place.

---

## 12. End-to-End Workflows

### Workflow 1: Critical Drift Detection → Retrain → HIL Approval

**Step 1: Dashboard sends predictions (user clicks "Critical Drift")**

```
dashboard/app.py: _run_drift_demo("Critical", 100, 400, CRITICAL_SHIFT)
  → For i in 0..499:
      POST /predict/ with NORMAL_PAYLOAD (if i < 100) or CRITICAL_SHIFT (if i >= 100)
  → GET /drift/report
```

Each prediction call: `platform/app/routers/predict.py:predict()`
- Build DataFrame → add derived feature → `model.predict_proba()` → apply threshold → append to `app.state.drift_accumulator`

**Step 2: Platform computes drift**

`platform/app/routers/drift.py:get_report()`
- 500 records in accumulator → proceed
- Split: reference[0:250], current[250:500]
- PSI on 9 numeric features (euribor3m scores ~0.30 — above critical threshold 0.25)
- Chi2 on 10 categorical features
- Output drift: PSI on probabilities
- Severity: critical (max PSI >= 0.25)
- `last_severity` was "stable", now "critical" → emit webhook
- POST to `http://agent:8001/webhook/drift`

**Step 3: Agent receives webhook**

`agent/app/routers/webhook.py:receive_drift_alert()`
- Pydantic validates DriftAlert
- `run_investigation(body)` → LangGraph flow

**Step 4: LangGraph triage**

`agent/app/graph/run_triage.py:run_triage()`
- Severity: critical
- Calls Azure OpenAI for summary (or uses deterministic fallback)
- System prompt: "Do not recommend production changes"

**Step 5: LangGraph action mapping**

`agent/app/graph/run_action.py:run_action()`
- critical → recommended_action = "retrain", status = "open"

**Step 6: LangGraph execute action**

`agent/app/graph/run_execute_action.py:run_execute_action()`
- action = "retrain" → NOT in PRODUCTION_ACTIONS
- `dispatch_retrain()` → `enqueue_job(job_type="retrain", ...)`
- `SADD` idempotency check → `RPUSH` to queue
- State: queued=true, status="queued"

**Step 7: Worker picks up job**

`worker/app/worker/consume_queue.py:process_job()`
- `BLPOP` from `drift-triage-jobs`
- `SETNX` idempotency check
- `handle_retrain()`:
  - `run_training_pipeline()` in executor thread
  - 5-fold CV → threshold = 0.3493
  - Train full pipeline → log to MLflow → register as candidate
  - Save model.joblib to disk
  - Fetch metrics from MLflow
- `POST /hil/notify-candidate` → agent creates HIL approval

**Step 8: Dashboard shows approval**

`dashboard/app.py` reads `GET /hil/pending` and shows the newest pending approval card with:
- Action: `promote_candidate`
- Target: the concrete candidate version resolved by the worker
- Approver input and approve/reject actions
- Registry and queue context beside the inbox

**Step 9: Human approves, promotes**

- User clicks "Approve" in HIL Inbox
- Dashboard: `POST /hil/{id}/approve` -> status = `approved`
- Agent dispatches `POST /registry/promote`
- Platform validates the approved `promote_candidate` row, writes the audit row, changes the MLflow alias, and reloads the model
- Registry Status now shows the updated Production version and metrics

### Workflow 2: Stable Drift (no action)

**Step 1: Dashboard sends 500 identical predictions**

**Step 2: Platform computes drift**
- All PSI values ≈ 0 (distributions are identical)
- Severity: stable
- `last_severity` was "stable", now "stable" → webhook SUPPRESSED
- Response: `webhook_sent: false, webhook_error: "severity unchanged — webhook suppressed"`

**Why suppress?** No change in severity means no new information. The agent already knows the system is stable. A suppressed webhook is correct behavior, not a failure.

### Workflow 3: Rollback

**Precondition:** a Production version exists and `previous_production_version` is available from promotion audit history

**Step 1: User creates a rollback HIL approval**
- Could be via agent dispatch or manual creation

**Step 2: User approves the rollback approval**
- `POST /hil/{id}/approve` → status = "approved"

**Step 3: User initiates rollback from dashboard**
- Opens Promotion History expander
- Enters rollback approval ID
- Clicks the rollback button for the previously audited version
- Dashboard: `POST /registry/rollback {target_version: "<previous>", approval_id: "...", approved_by: "admin"}`

**Step 4: Platform validates and executes**
- `_validate_rollback_approval()`: checks approval exists, is approved, is for rollback, version matches
- `_capture_current_production()`: captures the current Production version as `previous_version`
- `_write_rollback_audit()`: INSERT into `promotion_audit` before alias mutation
- `mlflow_client.set_registered_model_alias("Production", "<target_version>")`
- `_reload_model()`: loads the new Production alias from MLflow
- Returns updated status

**Step 5: Dashboard refreshes**
- `production_version` now reflects the rollback target
- `previous_production_version` now reflects the version that was active before the rollback
- promotion history contains the new rollback audit row

---

## 13. Safety Model

### The Safety Invariant

**Production is NEVER changed automatically.**

Every path to changing the `Production` alias in MLflow requires:
1. A HIL approval record in Postgres with `status = "approved"`
2. For rollback: the approval's `requested_action` must be `"rollback"` and `target_model_version` must match
3. For promotion: the `approved_by` field must be non-empty and the promotion checklist must pass

### What CAN change Production

| Action | Trigger | HIL Required? | Audit Recorded? |
|--------|---------|---------------|-----------------|
| `POST /registry/promote` | Dashboard or API | Yes (approved_by required) | Yes (promotion_audit) |
| `POST /registry/rollback` | Dashboard or API | Yes (approved HIL approval required) | Yes (promotion_audit) |

### What CANNOT change Production

| Action | Why not |
|--------|---------|
| Retraining | Creates candidate model only, sets `candidate` alias |
| Replay test | Read-only — loads model, predicts, logs results |
| Agent LangGraph flow | `run_execute_action.py` gating: Production actions create HIL approval, never dispatch |
| Critical drift alert | Agent maps to `retrain` → worker runs retrain → candidate created, not Production |
| LLM output | LLM only writes summaries. Action routing is deterministic and hardcoded |
| Dashboard drift buttons | Only call `/predict/` and `/drift/report` — no model mutation |

### What could go wrong and how it's prevented

| Failure mode | Prevention |
|-------------|-----------|
| Duplicate webhook → duplicate retrain | Agent: `SADD` idempotency check. Worker: `SETNX` dedup. Postgres: UNIQUE idempotency_key |
| Worker crashes mid-retrain | Redis: job not acknowledged. `SETNX` prevents re-dispatch. Retry + DLQ on next poll |
| Postgres down during promotion | Audit write fails → MLflow alias change aborted. Platform returns 500 |
| MLflow down during promotion | Step 5 (set alias) fails → audit already written. Platform returns 500. Audit row exists but alias unchanged — idempotent retry will succeed |
| LLM hallucinates "promote to Production" | LLM output is NEVER used for action routing. `run_action.py` is hardcoded. System prompt says "do not recommend production changes" |
| Rollback to wrong version | `_validate_rollback_approval()` checks `target_model_version` matches. Version must exist in MLflow |
| Rollback without approval | `_validate_rollback_approval()` returns 404/409. Dashboard disables button until approval ID entered |
| Model reload after rollback loads wrong model | `mlflow.sklearn.load_model("models:/...@Production")` loads from MLflow alias, which was just updated to the correct version |

### Audit Trail Completeness

Every Promotion and Rollback is recorded with:
- **What** version was changed (model_uri)
- **Who** approved it (approved_by)
- **When** it happened (timestamp)
- **Why** (linked to investigation_id)
- **From what** alias to what alias (from_alias → to_alias)
- **What was Production before** (previous_version) — for rollback

The audit trail in Postgres is append-only — rows are never updated or deleted. This makes it tamper-evident and SOC2-compliant.

---

## 14. File Index

### Platform (`platform/`)
| File | Lines | Purpose |
|------|-------|---------|
| `app/main.py` | 47 | FastAPI app assembly, lifespan, router mounting |
| `app/dependencies.py` | 41 | Dependency injection for model, threshold, HTTP client |
| `app/config/settings.py` | 57 | Centralized pydantic-settings configuration |
| `app/routers/predict.py` | 90 | Single prediction endpoint, accumulator |
| `app/routers/drift.py` | 209 | PSI/chi2 computation, webhook emission, severity classification |
| `app/routers/registry.py` | 380 | Promote, rollback, status, history, model reload, approval validation |
| `app/routers/queue.py` | 43 | Redis queue status endpoint |
| `app/schemas/predict_request.py` | 29 | 19-feature Pydantic model |
| `app/schemas/predict_response.py` | 8 | Prediction + probability response |
| `app/schemas/promote_request.py` | 11 | Promote request model |
| `app/schemas/drift_report.py` | 12 | Internal drift report model |
| `app/services/run_training.py` | 279 | Full ML training pipeline (load, preprocess, CV, train, log, save) |
| `app/services/validate_promotion.py` | 99 | 5-check promotion checklist |
| `app/services/compute_drift.py` | 55 | Alternative drift computation (scipy-based) |

### Agent (`agent/`)
| File | Lines | Purpose |
|------|-------|---------|
| `app/main.py` | 22 | FastAPI app, router mounting |
| `app/graph/state.py` | 27 | AgentState TypedDict |
| `app/graph/build_graph.py` | 128 | StateGraph construction, supervisor topology |
| `app/graph/run_supervisor.py` | 51 | Deterministic routing between nodes |
| `app/graph/run_triage.py` | 54 | Severity → summary (LLM with deterministic fallback) |
| `app/graph/run_action.py` | 25 | Severity → action (purely deterministic) |
| `app/graph/run_execute_action.py` | 86 | Dispatch or create HIL approval |
| `app/graph/run_comms.py` | 59 | Final summary (LLM with fallback) |
| `app/routers/webhook.py` | 32 | DriftAlert webhook endpoint |
| `app/routers/hil.py` | 221 | HIL approval CRUD endpoints |
| `app/schemas/drift_alert.py` | 71 | Versioned webhook schema |
| `app/schemas/hil_action.py` | 50 | Approval request/decision/response models |
| `app/schemas/investigation.py` | 40 | Status machine, action literals |
| `app/services/request_approval.py` | 245 | Postgres CRUD for HIL approvals, idempotency, status transitions |
| `app/services/manage_checkpoints.py` | 46 | Optional LangGraph Postgres saver helper |
| `app/services/investigations.py` | live repo file | Investigation and checkpoint persistence in Postgres |
| `app/tools/queue_client.py` | 112 | Redis enqueue with idempotency |
| `app/tools/dispatch_replay.py` | 34 | Builds replay job payload |
| `app/tools/dispatch_retrain.py` | 27 | Builds retrain job payload |
| `app/tools/dispatch_rollback.py` | 32 | Builds rollback job payload (requires approval_id) |
| `app/config/settings.py` | 38 | Agent config (LLM provider, tracing, etc.) |

### Worker (`worker/`)
| File | Lines | Purpose |
|------|-------|---------|
| `app/worker/consume_queue.py` | 365 | Redis queue consumer, retry/DLQ logic, retrain/replay/rollback handlers |

### Dashboard (`dashboard/`)
| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 364 | Streamlit operator dashboard |

### Infrastructure
| File | Lines | Purpose |
|------|-------|---------|
| `docker-compose.yml` | 177 | 8-service orchestration with healthchecks, volumes, networks |
| `postgres/init.sql` | 57 | Database schema (3 tables, 7 indexes) |
| `mlflow/Dockerfile` | 11 | MLflow server with artifact proxy |

---

## Final Words for the Code Review

If the instructor asks *"Why did you build it this way instead of X?"* — the answer is always traceable to one of these principles:

1. **Safety first** — No automatic Production mutation. HIL gates on every production change. Deterministic action routing (no LLM hallucination risk).

2. **Auditability** — Every promotion and rollback is recorded in Postgres with who, what, when, and previous version. The audit is durable (written before mutation).

3. **Separation of concerns** — 8 services, each with one responsibility. The agent decides but doesn't execute. The worker executes but doesn't decide. The dashboard displays but doesn't mutate models.

4. **Defense in depth** — Idempotency at multiple layers (Redis + Postgres). Fallbacks on every LLM call. Validation on every production change. Multiple safety checks on rollback.

5. **Testability** — Zero global state. Dependency injection everywhere. Lazy imports for optional dependencies. Every function is independently testable.
