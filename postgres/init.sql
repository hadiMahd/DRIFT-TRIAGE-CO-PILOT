CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    drift_event_id TEXT NOT NULL,
    status TEXT NOT NULL,
    severity TEXT NOT NULL,
    recommended_action TEXT NULL,
    summary TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hil_approvals (
    approval_id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    drift_event_id TEXT NOT NULL,
    requested_action TEXT NOT NULL,
    target_model_version TEXT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    requested_by TEXT NOT NULL DEFAULT 'agent',
    approved_by TEXT NULL,
    reason TEXT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_hil_approvals_status
    ON hil_approvals (status);

CREATE INDEX IF NOT EXISTS idx_hil_approvals_investigation_id
    ON hil_approvals (investigation_id);

CREATE INDEX IF NOT EXISTS idx_hil_approvals_drift_event_id
    ON hil_approvals (drift_event_id);

CREATE INDEX IF NOT EXISTS idx_investigations_status
    ON investigations (status);

CREATE INDEX IF NOT EXISTS idx_investigations_drift_event_id
    ON investigations (drift_event_id);

CREATE TABLE IF NOT EXISTS investigation_checkpoints (
    investigation_id TEXT PRIMARY KEY,
    drift_event_id TEXT NOT NULL,
    last_completed_node TEXT NULL,
    state_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_investigation_checkpoints_drift_event_id
    ON investigation_checkpoints (drift_event_id);

CREATE TABLE IF NOT EXISTS promotion_audit (
    id SERIAL PRIMARY KEY,
    model_uri TEXT NOT NULL,
    investigation_id TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    from_alias TEXT NULL,
    to_alias TEXT NOT NULL DEFAULT 'Production',
    previous_version TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_promotion_audit_model_uri
    ON promotion_audit (model_uri);

CREATE INDEX IF NOT EXISTS idx_promotion_audit_timestamp
    ON promotion_audit (timestamp DESC);

CREATE TABLE IF NOT EXISTS platform_drift_state (
    state_id BOOLEAN PRIMARY KEY DEFAULT TRUE,
    drift_accumulator JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_severity TEXT NOT NULL DEFAULT 'stable',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (state_id = TRUE)
);

INSERT INTO platform_drift_state (state_id, drift_accumulator, last_severity)
VALUES (TRUE, '[]'::jsonb, 'stable')
ON CONFLICT (state_id) DO NOTHING;
