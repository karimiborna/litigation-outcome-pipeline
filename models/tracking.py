"""MLflow tracking helpers for experiment logging and model registration."""

from __future__ import annotations

import logging
from typing import Any

from mlflow.tracking import MlflowClient

import mlflow
from models.config import MLflowConfig

logger = logging.getLogger(__name__)

# Appended when an experiment was created with a server-local artifact path; remote
# clients cannot mkdir e.g. /home/user/... on their laptops (needs --serve-artifacts
# on the server *and* experiments created after that, or a fresh experiment name).
_REMOTE_SAFE_EXPERIMENT_SUFFIX = "-remote-artifacts"


def _tracking_uri_is_remote_http(tracking_uri: str) -> bool:
    u = tracking_uri.lower().strip()
    return u.startswith("http://") or u.startswith("https://")


def _artifact_location_breaks_remote_clients(
    artifact_location: str | None,
    tracking_uri: str,
) -> bool:
    """Remote clients must not use server-only paths (e.g. /home/...) for artifacts."""
    if not _tracking_uri_is_remote_http(tracking_uri):
        return False
    loc = (artifact_location or "").strip()
    if not loc:
        return False
    lower = loc.lower()
    if lower.startswith("file:"):
        return True
    # Backend sometimes stores absolute POSIX paths without file: — still server-local.
    if loc.startswith("/"):
        return True
    return False


def init_mlflow(config: MLflowConfig | None = None) -> MlflowClient:
    """Initialize MLflow with the given config and return a client."""
    config = config or MLflowConfig()
    mlflow.set_tracking_uri(config.tracking_uri)
    if config.registry_uri:
        mlflow.set_registry_uri(config.effective_registry_uri)
    return MlflowClient(tracking_uri=config.tracking_uri)


def get_or_create_experiment(name: str, config: MLflowConfig | None = None) -> str:
    """Get an existing experiment by name or create it. Returns experiment ID."""
    config = config or MLflowConfig()
    mlflow.set_tracking_uri(config.tracking_uri)

    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        if _artifact_location_breaks_remote_clients(
            experiment.artifact_location,
            config.tracking_uri,
        ):
            if name.endswith(_REMOTE_SAFE_EXPERIMENT_SUFFIX):
                raise RuntimeError(
                    f"Experiment {name!r} still has a server-local artifact_location "
                    f"({experiment.artifact_location!r}). On the MLflow server use "
                    "--serve-artifacts and --artifacts-destination, then delete this "
                    "experiment in the UI and retry."
                )
            safe_name = name + _REMOTE_SAFE_EXPERIMENT_SUFFIX
            logger.warning(
                "Experiment %r uses server-local artifact_location=%r; "
                "using %r instead (old experiment can be deleted in the MLflow UI).",
                name,
                experiment.artifact_location,
                safe_name,
            )
            return get_or_create_experiment(safe_name, config)
        return experiment.experiment_id

    # Remote HTTP tracking: let the server pick artifact storage (required with
    # --serve-artifacts). Passing mlruns/artifacts here becomes a bad client-local path.
    kwargs: dict[str, Any] = {}
    if not _tracking_uri_is_remote_http(config.tracking_uri):
        kwargs["artifact_location"] = config.artifact_root

    experiment_id = mlflow.create_experiment(name, **kwargs)
    logger.info("Created MLflow experiment '%s' (id=%s)", name, experiment_id)
    return experiment_id


def start_run(
    experiment_name: str,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    config: MLflowConfig | None = None,
) -> mlflow.ActiveRun:
    """Start an MLflow run in the given experiment, optionally logging params."""
    config = config or MLflowConfig()
    experiment_id = get_or_create_experiment(experiment_name, config)

    run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    if params:
        mlflow.log_params(params)
    return run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dictionary of metrics to the active run."""
    mlflow.log_metrics(metrics, step=step)


def log_model_artifact(
    model: Any,
    artifact_path: str,
    registered_name: str | None = None,
) -> None:
    """Log a sklearn model artifact and optionally register it."""
    mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_name,
    )


def transition_model_stage(
    model_name: str,
    version: int,
    stage: str,
    config: MLflowConfig | None = None,
) -> None:
    """Transition a registered model version to a new stage.

    Stages: None, Staging, Production, Archived.
    """
    client = init_mlflow(config)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
    )
    logger.info("Model %s v%d transitioned to %s", model_name, version, stage)


def load_production_model(model_name: str, config: MLflowConfig | None = None) -> Any:
    """Load the Production-stage model from the registry."""
    config = config or MLflowConfig()
    mlflow.set_tracking_uri(config.tracking_uri)
    model_uri = f"models:/{model_name}/Production"
    return mlflow.sklearn.load_model(model_uri)
