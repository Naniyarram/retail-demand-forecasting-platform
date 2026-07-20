"""
experiment_runner.py

Forecasting experimentation framework.

Supports:
- Walk Forward Validation
- Multi Model Comparison
- Leaderboard Generation
- Champion Selection

Future:
- MLflow Tracking
- Model Registry
- CI/CD

"""
from pipeline.training.mlflow_manager import (
    MLflowManager
)
from typing import List
from typing import Dict
from typing import Any

import pandas as pd
import numpy as np

from pipeline.config.settings import (
    FORECAST_HORIZON,
    PRIMARY_METRIC
)

from pipeline.evaluation.walk_forward import (
    WalkForwardValidator
)


class ExperimentRunner:
    """
    Multi-model experimentation framework.
    """

    def __init__(
        self,
        models,
        forecast_horizon=FORECAST_HORIZON,
        primary_metric=PRIMARY_METRIC,
        mlflow_manager: MLflowManager | None = None
    ):

        self.models = models

        self.forecast_horizon = forecast_horizon

        self.primary_metric = primary_metric

        self.mlflow_manager = mlflow_manager

        self.validator = WalkForwardValidator(
            forecast_horizon=forecast_horizon
        )

        self.results = []

        self.leaderboard = None

    def run_model(
        self,
        model,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation
        for a single model.
        """

        model_name = model.get_model_name()
        print(f"\n[INFO] Running walk-forward validation for: {model_name}")

        fold_metrics = []

        for fold_number, (
            train_idx,
            test_idx
        ) in enumerate(
            self.validator.split(data),
            start=1
        ):
            print(f"  +- Fold {fold_number}/{self.validator.n_splits} - Training and evaluating...")

            train_df = (
                data.iloc[train_idx]
                .copy()
            )

            test_df = (
                data.iloc[test_idx]
                .copy()
            )

            model.fit(train_df)

            predictions = model.predict(
                horizon=len(test_df)
            )

            metrics = model.evaluate(
                y_true=test_df["Weekly_Sales"],
                y_pred=predictions
            )

            metrics["fold"] = fold_number

            fold_metrics.append(
                metrics
            )

        fold_df = pd.DataFrame(
            fold_metrics
        )

        result = {

            "model_name":
                model_name,

            "RMSE":
                fold_df["RMSE"].mean(),

            "MAE":
                fold_df["MAE"].mean(),

            "MAPE":
                fold_df["MAPE"].mean(),

            "fold_results":
                fold_df,

            "parameters":
                model.get_params(),
                
            "model_object": model,
        }

        print(f"  +- Completed: Mean MAPE = {result['MAPE']:.4f} | Mean RMSE = {result['RMSE']:.4f}")


        if self.mlflow_manager:

            with self.mlflow_manager.start_run(
            run_name=model.get_model_name(),
            nested=True
            ):

                self.mlflow_manager.log_params(
                    model.get_params()
                )

                self.mlflow_manager.log_metrics(
                    {
                        "RMSE": result["RMSE"],
                        "MAE": result["MAE"],
                        "MAPE": result["MAPE"]
                    }
                )

                self.mlflow_manager.set_tags({ "model_type":model.get_model_name() } )
        return result

    def run(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run all models.
        """

        self.results = []

        if self.mlflow_manager:

            with self.mlflow_manager.start_run(
                run_name="forecasting_experiment"
            ):

                for model in self.models:

                    result = self.run_model(
                        model=model,
                        data=data
                    )

                    self.results.append(
                        result
                    )

                leaderboard = pd.DataFrame(
                    [
                        {
                            "Model":
                                result["model_name"],

                            "RMSE":
                                result["RMSE"],

                            "MAE":
                                result["MAE"],

                            "MAPE":
                                result["MAPE"]
                        }
                        for result in self.results
                    ]
                )

                leaderboard = (
                    leaderboard
                    .sort_values(
                        self.primary_metric
                    )
                    .reset_index(
                        drop=True
                    )
                )

                leaderboard.insert(
                    0,
                    "Rank",
                    range(
                        1,
                        len(leaderboard) + 1
                    )
                )

                self.leaderboard = leaderboard

                self.mlflow_manager.log_leaderboard(
                    leaderboard
                )

                champion = (
                    self.get_champion()
                )

                self.mlflow_manager.log_champion(
                    champion
                )

                return leaderboard

        # fallback when MLflow disabled

        for model in self.models:

            result = self.run_model(
                model=model,
                data=data
            )

            self.results.append(
                result
            )

        leaderboard = pd.DataFrame(
            [
                {
                    "Model":
                        result["model_name"],

                    "RMSE":
                        result["RMSE"],

                    "MAE":
                        result["MAE"],

                    "MAPE":
                        result["MAPE"]
                }
                for result in self.results
            ]
        )

        leaderboard = (
            leaderboard
            .sort_values(
                self.primary_metric
            )
            .reset_index(
                drop=True
            )
        )

        leaderboard.insert(
            0,
            "Rank",
            range(
                1,
                len(leaderboard) + 1
            )
        )

        self.leaderboard = leaderboard

        return leaderboard
    
    def get_leaderboard(
        self
    ) -> pd.DataFrame:

        if self.leaderboard is None:

            raise RuntimeError(
                "Run experiments first."
            )

        return self.leaderboard

    def get_champion(
        self
    ) -> Dict[str, Any]:
        """
        Return best model.
        """

        if not self.results:

            raise RuntimeError(
                "Run experiments first."
            )

        champion = min(
            self.results,
            key=lambda x:
            x[self.primary_metric]
        )

        return champion

    def get_full_results(
        self
    ) -> List[Dict]:
        """
        Return detailed experiment results.
        """

        return self.results