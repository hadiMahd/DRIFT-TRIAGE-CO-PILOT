"""Promotion gate — programmatic checklist before accepting a promote request.

Day-4 checklist assertions:
1. model_uri exists in MLflow registry
2. Model artifacts include schema.json
3. Model artifacts include model_card.json with md5 hash and environment fingerprint
4. Test metrics meet minimum bar (recall >= 0.75)
5. Model has "candidate" alias (must go through candidate stage before Production)

Raises ValueError with descriptive message if any check fails.
Called by routers/registry.py.
"""

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from app.config.settings import Settings


def parse_model_reference(model_uri: str) -> tuple[str, str | None, str | None]:
    clean = model_uri.removeprefix("models:/").strip()
    if "/" in clean:
        name, version = clean.rsplit("/", 1)
        return name, version, None
    if "@" in clean:
        name, alias = clean.split("@", 1)
        return name, None, alias
    return clean, None, None


def assert_promotion_checklist(model_uri: str) -> None:
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    model_name, requested_version, requested_alias = parse_model_reference(model_uri)

    if model_name != settings.registered_model_name:
        raise ValueError(
            f"Requested model '{model_name}' does not match registered model "
            f"'{settings.registered_model_name}'."
        )

    try:
        candidate = client.get_model_version_by_alias(
            name=settings.registered_model_name,
            alias="candidate",
        )
    except MlflowException:
        raise ValueError(
            f"No model found at '{settings.registered_model_name}@candidate'. "
            "Retrain first to produce a candidate."
        )

    if requested_version is not None:
        version = requested_version
        try:
            mv = client.get_model_version(name=settings.registered_model_name, version=version)
        except MlflowException as exc:
            raise ValueError(f"Requested model version {version} was not found.") from exc
    elif requested_alias is not None:
        try:
            mv = client.get_model_version_by_alias(
                name=settings.registered_model_name,
                alias=requested_alias,
            )
        except MlflowException as exc:
            raise ValueError(
                f"No model found at '{settings.registered_model_name}@{requested_alias}'."
            ) from exc
        version = str(mv.version)
    else:
        raise ValueError("model_uri must reference an exact version or a registered alias.")

    if str(candidate.version) != str(version):
        raise ValueError(
            f"Requested version {version} is not the current candidate alias "
            f"(candidate={candidate.version})."
        )

    run_id = mv.run_id
    if not run_id:
        raise ValueError(f"Model version {version} has no associated run. Cannot validate artifacts.")

    # 2 — schema.json exists
    try:
        schema_path = client.download_artifacts(run_id, "schema.json")
        if not schema_path:
            raise ValueError("schema.json artifact is empty or missing.")
    except Exception:
        raise ValueError(
            f"schema.json not found for model version {version}. "
            "Ensure run_training_pipeline() logs the schema artifact."
        )

    # 3 — model_card.json with md5 and environment fingerprint
    try:
        card_path = client.download_artifacts(run_id, "model_card.json")
        if not card_path:
            raise ValueError("model_card.json artifact is empty or missing.")
    except Exception:
        raise ValueError(
            f"model_card.json not found for model version {version}. "
            "Ensure run_training_pipeline() logs the model_card artifact."
        )

    import json
    with open(card_path) as f:
        card = json.load(f)

    if "dataset" not in card or "sha256" not in card.get("dataset", {}):
        raise ValueError("model_card.json missing dataset.sha256 hash.")

    if "environment" not in card:
        raise ValueError("model_card.json missing environment fingerprint.")

    # 4 — metrics meet bar
    metrics = client.get_run(run_id).data.metrics
    recall = metrics.get("test_recall")
    if recall is None:
        raise ValueError("test_recall metric not found in run. Was training logged correctly?")
    if recall < settings.min_recall:
        raise ValueError(
            f"test_recall={recall:.4f} is below minimum {settings.min_recall}. "
            "Model does not meet recall bar."
        )

    # 5 — already has candidate alias (verified in step 1)
    print(f"Promotion gate passed for version {version}. Recall={recall:.4f}")
