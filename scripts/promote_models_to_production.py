#!/usr/bin/env python3
"""Transition the latest registry version of each model to Production."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/promote_models_to_production.py` without editable install
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlflow.tracking import MlflowClient

from models.config import MLflowConfig
from models.tracking import init_mlflow


def _promote_latest(client: MlflowClient, name: str) -> int:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"No registered versions found for model '{name}'")
    latest = max(versions, key=lambda v: int(v.version))
    vid = int(latest.version)
    client.transition_model_version_stage(name=name, version=vid, stage="Production")
    return vid


def main() -> int:
    config = MLflowConfig()
    client = init_mlflow(config)
    c_ver = _promote_latest(client, config.classifier_model_name)
    r_ver = _promote_latest(client, config.regressor_model_name)
    print(f"Production: {config.classifier_model_name} v{c_ver}")
    print(f"Production: {config.regressor_model_name} v{r_ver}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
