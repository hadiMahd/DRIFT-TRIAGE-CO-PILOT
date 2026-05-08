# Decisions

## Platform Decisions (Hadi)

### Model Selection: HistGradientBoostingClassifier

Three models were compared via stratified cross-validation and HistGradientBoostingClassifier was chosen because it performed best overall while staying fast and sklearn-native.

### Threshold Rule: Highest Threshold Where Recall >= 0.75

The operating threshold follows the Week 5 requirement:

`precision_recall_curve(y_true, y_proba) -> highest threshold with recall >= 0.75`

That threshold is frozen after validation and then applied to test and serving.

### Artifact Hashing: SHA256

Model artifacts are hashed with SHA256 and the hash is carried through the model card and validation checks.

### Candidate Alias Pattern

Training registers a candidate model version only. Production is never set automatically during training.

### Feature Preprocessing

- drop `duration`
- preserve `unknown` as a real category
- turn `pdays == 999` into a sentinel feature
- keep preprocessing inside the sklearn pipeline

### Worker Idempotency

Worker-side idempotency follows:

`idempotency:{action}:{investigation_id}:{target}`

### Promotion Audit

Successful promotions write an audit row to Postgres.

### Rollback Safety

Rollback requires `approval_id` and remains intentionally non-automatic. Current worker behavior DLQs rollback jobs rather than mutating Production directly.

## Agent Decisions (Jad)

### Deterministic Safety Baseline

Deterministic action rules remain the source of truth even when LLM support is enabled:

- stable -> `none`
- moderate -> `replay_test`
- critical -> `retrain`

The agent does not let model output bypass these rules.

### Optional LLM Mode

`LLM_PROVIDER=mock` is the default and requires no API keys.

`LLM_PROVIDER=azure` enables Azure-hosted Kimi usage through environment variables:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT` or `AZURE_STRONG_MODEL`

The strong-model default is:

`AZURE_STRONG_MODEL=Kimi-K2.6-1`

LLM output may improve summaries or rationale, but it cannot directly execute a production-changing action.

### LangGraph Wrapper

The agent now runs through a LangGraph `StateGraph` wrapper:

`triage -> action -> execute_action -> comms`

This preserves the existing deterministic flow while giving the project a real graph wrapper for later expansion.

### Safe Action Policy

Safe actions are queued:

- `replay_test`
- `retrain`

Production-impacting actions are gated:

- `rollback` requires HIL approval
- `promote_candidate` requires HIL approval

The agent never promotes or rolls back Production directly.

### Redis Queue Contract

Shared queue contract:

- queue: `drift-triage-jobs`
- DLQ: `DLQ:drift-triage-jobs`
- idempotency format: `idempotency:{action}:{investigation_id}:{target_or_event}`

The dashboard only displays queue state; it does not own queue logic.

### Postgres Persistence

The agent persists HIL approvals and related state in Postgres. This persistence is the current recovery-critical storage.

### LangGraph Checkpoint Status

Current implementation persists HIL approvals and approval state in Postgres. LangGraph checkpoint integration is prepared but not used as the main recovery mechanism yet.

### Stable Drift Cost Control

Stable drift does not require an expensive LLM decision to preserve responsiveness and keep the demo usable without external model calls.

## Infrastructure Decisions (Shared)

### Docker Service-Name Networking

Inside Compose, services communicate by service name instead of `localhost`.

### Default Demo Mode

The clean no-key demo path uses:

- `LLM_PROVIDER=mock`
- Docker Compose service discovery
- Redis + Postgres + MLflow local containers

### Dashboard Role

The dashboard is a control room and operator UI. It displays service health, queue state, registry state, and HIL approvals, but it does not become the source of business logic.

### Postgres Scope

Postgres stores:

- `hil_approvals`
- `promotion_audit`

It is not yet the primary LangGraph checkpoint store in production practice, even though the helper exists.

### Safety Boundary

Human approval remains the boundary for any action that could change Production.

## Open Questions

- Webhook vs polling: webhook remains the primary drift handoff mechanism for now.
- LLM choice: Azure-hosted Kimi is supported, but mock mode remains the default demo-safe setting.
- Queue idempotency strategy: Redis idempotency key plus worker retry/DLQ behavior.
- HIL stale-approval handling: expiry and escalation rules are still to be formalized.
- Checkpoint store sync with registry: future work once LangGraph checkpoint resume becomes a first-class recovery path.
