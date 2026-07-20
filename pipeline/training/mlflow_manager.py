"""
mlflow_manager.py

Centralized MLflow tracking layer.

Responsibilities:
- Experiment Tracking
- Metric Logging
- Parameter Logging
- Artifact Logging

Future:
- Model Registry
- Champion Promotion

"""

from pathlib import Path
from typing import Dict
from typing import Any
from typing import Optional

import mlflow
import pandas as pd


class MLflowManager:

    def __init__(
        self,
        experiment_name: str
    ):
        import os
        if not os.getenv("MLFLOW_TRACKING_URI"):
            tracking_dir = Path(
                "artifacts/mlflow_runs"
            )

            tracking_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            mlflow.set_tracking_uri(
                tracking_dir.resolve().as_uri()
            )

        self.experiment_name = (
            experiment_name
        )

        mlflow.set_experiment(
            experiment_name
        )

    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False
    ):

        return mlflow.start_run(
            run_name=run_name,
            nested=nested
        )

    def log_params(
        self,
        params: Dict[str, Any]
    ):

        mlflow.log_params(
            params
        )

    def log_metrics(
        self,
        metrics: Dict[str, float]
    ):

        mlflow.log_metrics(
            metrics
        )

    def log_leaderboard(
        self,
        leaderboard: pd.DataFrame,
        artifact_path: str = "leaderboard"
    ):

        output_path = (
            Path(
                "artifacts/leaderboards"
            )
            / "leaderboard.csv"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        leaderboard.to_csv(
            output_path,
            index=False
        )

        mlflow.log_artifact(
            str(output_path),
            artifact_path=artifact_path
        )

    def log_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        artifact_path: str
    ):

        output_path = (
            Path("artifacts")
            / filename
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
        )

        mlflow.log_artifact(
            str(output_path),
            artifact_path=artifact_path
        )

    def log_champion(
        self,
        champion: Dict[str, Any]
    ):

        champion_metrics = {

            "champion_rmse":
                champion["RMSE"],

            "champion_mae":
                champion["MAE"],

            "champion_mape":
                champion["MAPE"]
        }

        mlflow.log_metrics(
            champion_metrics
        )

        mlflow.set_tag(
            "champion_model",
            champion["model_name"]
        )

    def set_tags(
        self,
        tags: Dict[str, str]
    ):

        mlflow.set_tags(
            tags
        )

    def end_run(
        self
    ):

        mlflow.end_run()
