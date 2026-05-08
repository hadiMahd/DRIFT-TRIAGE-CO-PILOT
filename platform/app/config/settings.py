"""Platform configuration via pydantic-settings.

All env vars read from .env file. extra="forbid" — unknown vars rejected.
No scattered os.getenv() anywhere else in the codebase.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PLATFORM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PLATFORM_ROOT.parent


def _resolve_platform_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PLATFORM_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # Default: Docker service name. Override via .env to http://localhost:5000 for local dev.
    mlflow_tracking_uri: str = "http://mlflow:5000"
    agent_base_url: str = "http://agent:8001"
    redis_url: str = "redis://redis:6379/0"
    postgres_dsn: str = "postgresql+asyncpg://user:pass@postgres:5432/drift"
    registered_model_name: str = "bank_marketing_pipeline"
    model_path: str = "data/model.joblib"
    dataset_path: str = "data/bank-additional-full.csv"
    threshold: float = 0.3493
    min_recall: float = 0.75
    cv_folds: int = 5
    drift_window_size: int = 500
    drift_severity_moderate: float = 0.10
    drift_severity_critical: float = 0.25

    def resolved_model_path(self) -> Path:
        return _resolve_platform_path(self.model_path)

    def resolved_dataset_path(self) -> Path:
        configured = _resolve_platform_path(self.dataset_path)
        if configured.exists():
            return configured

        fallback = REPO_ROOT / "initial-training" / "dataset" / "bank-additional-full.csv"
        if fallback.exists():
            return fallback

        return configured
