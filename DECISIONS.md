# Decisions

## Platform Decisions

### Model Selection: HistGradientBoostingClassifier

The serving model remains `HistGradientBoostingClassifier` because it performed best in the repo's training pipeline while staying sklearn-native and simple to ship in Docker.

### Threshold Rule: Highest Threshold Where Recall >= 0.75

The threshold-selection logic is unchanged:

`precision_recall_curve(y_true, y_proba) -> highest threshold with recall >= 0.75`

The numeric threshold may change after retraining, but the rule does not.

### Candidate Alias Pattern

Training only registers a candidate model version. Retraining does not move the `Production` alias.

### Registry Safety

- `GET /registry/status` exposes registered model name, production version, candidate version, previous production version, and production metrics when available.
- `GET /registry/history` returns promotion and rollback audit history from Postgres.
- `POST /registry/promote` requires `approval_id` and validates an approved `promote_candidate` HIL row before changing MLflow aliases.
- `POST /registry/rollback` requires `approval_id` and validates an approved rollback HIL row before changing MLflow aliases.

### Audit-First Mutation

Promotion and rollback write durable Postgres audit rows before mutating the MLflow `Production` alias. If the audit write fails, the alias change is aborted.

### Drift-State Persistence

Platform drift history is persisted in Postgres so prediction-window state and last severity survive service restarts.

## Agent Decisions

### Deterministic Routing And Action Policy

The supervisor topology is deterministic:

`START -> supervisor -> triage -> supervisor -> action -> supervisor -> execute_action -> supervisor -> comms -> END`

Severity-to-action mapping is also deterministic:

- `stable -> none`
- `moderate -> replay_test`
- `critical -> retrain`

LLM output can improve summaries, but it does not control routing or Production-changing decisions.

### HIL As The Production Boundary

The agent never mutates Production directly during webhook handling.

- Safe actions are queued immediately: `replay_test`, `retrain`
- Production-changing actions are gated: `promote_candidate`, `rollback`

Approving a HIL action dispatches the corresponding platform call from the agent:

- approve promotion -> `POST /registry/promote`
- approve rollback -> `POST /registry/rollback`

### Durable Investigation State

The agent persists:

- `investigations`
- `investigation_checkpoints`
- `hil_approvals`

Investigation state is reused by `drift_event_id`, so repeated delivery of the same drift event resumes the same investigation instead of creating a new one.

### Postgres-Backed Checkpointing

The live recovery path is Postgres-backed persistence owned by the repo:

- investigation rows
- checkpoint rows
- HIL approval rows

The optional LangGraph Postgres checkpoint helper still exists, but the proven recovery path in this project is the repo's own persisted state flow.

## Worker Decisions

### Queue Contract

Shared queue contract:

- queue: `drift-triage-jobs`
- DLQ: `DLQ:drift-triage-jobs`
- idempotency format: `idempotency:{action}:{investigation_id}:{target_or_event}`

### Candidate Notification Reliability

After retraining, the worker resolves the concrete candidate version and notifies the agent through `/hil/notify-candidate`.

That notification now uses retries and a longer timeout so retrain completion does not silently lose the HIL approval step.

### Rollback Handling

The worker rollback handler is implemented and sends:

- `target_model_version`
- `approval_id`
- `approved_by`

Missing `approval_id` is rejected safely.

## Infrastructure Decisions

### Docker Service-Name Networking

Inside Compose, services communicate by service name, not `localhost`.

### Demo Defaults

The no-key path uses:

- `LLM_PROVIDER=mock`
- local Compose networking
- Redis, Postgres, MLflow, and pgAdmin containers

### Dashboard Role

The dashboard is an operator console. It shows health, drift results, queue status, registry status, HIL approvals, and rollback controls, but it is not the source of business logic.

### Postgres Scope

Postgres stores:

- `hil_approvals`
- `investigations`
- `investigation_checkpoints`
- `platform_drift_state`
- `promotion_audit`

## Open Questions

- Whether to adopt the official LangGraph Postgres checkpoint backend instead of the current repo-owned persistence approach
- Whether to formalize expiry/escalation rules for long-lived pending HIL approvals
- Whether to automate a richer rollback-job initiation flow instead of relying on explicit approval-driven operator actions
