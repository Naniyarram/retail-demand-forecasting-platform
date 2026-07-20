"""
model_logger.py

Model artifact logging layer.

Responsibilities
----------------
- Log champion model artifacts
- Log model metadata
- Return model URI

"""

from typing import Any
from typing import Dict

import mlflow


from pipeline.training.forecasting_pyfunc import (
    ForecastingPyFuncModel
)


class ModelLogger:
    """
    Handles MLflow model logging.
    """

    def __init__(self):
        pass

    def log_model(
        self,
        model_name: str,
        model_object,
        metadata: dict | None = None,
        artifact_path: str = "champion_model"
    ) -> str:
        """
        Log production model
        as MLflow PyFunc model.
        """

        wrapped_model = (
            ForecastingPyFuncModel(
                forecasting_model=model_object
            )
        )

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=wrapped_model
        )

        mlflow.log_dict(
            {
                "model_name": model_name,
                "model_type":
                    type(model_object).__name__
            },
            f"{artifact_path}/model_info.json"
        )

        if metadata:

            mlflow.log_dict(
                metadata,
                f"{artifact_path}/metadata.json"
            )

        model_uri = (
            f"runs:/"
            f"{mlflow.active_run().info.run_id}"
            f"/{artifact_path}"
        )

        return model_uri

    def log_champion_metadata(
        self,
        champion_summary: Dict
    ) -> None:
        """
        Log champion metadata.
        """

        mlflow.log_dict(
            champion_summary,
            "champion_summary.json"
        )

    def create_model_uri(
        self,
        artifact_path: str = "champion_model"
    ) -> str:
        """
        Generate model URI.
        """

        active_run = mlflow.active_run()

        if active_run is None:

            raise RuntimeError(
                "No active MLflow run found."
            )

        return (
            f"runs:/"
            f"{active_run.info.run_id}"
            f"/{artifact_path}"
        )