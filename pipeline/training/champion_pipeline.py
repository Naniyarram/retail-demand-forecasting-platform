"""
champion_pipeline.py

Champion lifecycle orchestration.

Responsibilities
----------------
1. Champion Retraining
2. Champion Summary Creation
3. Model Logging
4. Model URI Generation


"""

from typing import Dict
from typing import Any

import pandas as pd
import mlflow

from pipeline.training.champion_trainer import (
    ChampionTrainer
)

from pipeline.training.model_logger import (
    ModelLogger
)

from pipeline.training.model_packager import (
    ModelPackager
)


class ChampionPipeline:
    """
    End-to-end champion workflow.
    """

    def __init__(self):

        self.trainer = ChampionTrainer()

        self.logger = ModelLogger()

        self.packager = ModelPackager()

    def run(
        self,
        champion_result: Dict[str, Any],
        full_dataset: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute champion workflow.

        Steps
        -----
        1. Retrain champion
        2. Generate summary
        3. Log metadata
        4. Generate model URI

        Returns
        -------
        dict
        """

        # ---------------------------------
        # Retrain Champion
        # ---------------------------------
        model_name = champion_result["model_name"]
        print(f"\n[INFO] Retraining champion model ({model_name}) on full dataset...")

        production_model = (
            self.trainer.retrain_champion(
                champion_result=champion_result,
                full_dataset=full_dataset
            )
        )

        # ---------------------------------
        # Generate Summary
        # ---------------------------------

        champion_summary = (
            self.trainer.get_champion_summary(
                champion_result
            )
        )

        # ---------------------------------
        # Log Metadata
        # ---------------------------------
        print(f"[INFO] Saving and packaging retrained champion model...")

        package_result = (
            self.packager.package_champion(
                model_name=champion_result[
                    "model_name"
                ],
                model_object=production_model,
                metadata=champion_summary
            )
        )

        # ---------------------------------
        # Create MLflow URI
        # ---------------------------------
        print(f"[INFO] Registering champion model in MLflow tracking server...")

        try:

            active_run = mlflow.active_run()

            if active_run is None:

                with mlflow.start_run(
                    run_name="champion_pipeline"
                ):

                    self.logger.log_champion_metadata(
                        champion_summary
                    )

                    model_uri = (
                        self.logger.log_model(
                            model_name=champion_result[
                                "model_name"
                            ],
                            model_object=production_model,
                            metadata=champion_summary
                        )
                    )

            else:

                self.logger.log_champion_metadata(
                    champion_summary
                )

                model_uri = (
                    self.logger.log_model(
                        model_name=champion_result[
                            "model_name"
                        ],
                        model_object=production_model,
                        metadata=champion_summary
                    )
                )

        except Exception as exc:

            print(
                "[WARNING] MLflow logging skipped: "
                f"{exc}"
            )

            model_uri = (
                "local:"
                f"{package_result['artifact_path']}"
            )

        return {

            "model_name":
                champion_result[
                    "model_name"
                ],

            "production_model":
                production_model,

            "champion_summary":
                champion_summary,

            "model_uri":
                model_uri,

            "artifact_path":
                package_result["artifact_path"],

            "metadata_path":
                package_result["metadata_path"]
        }

    def print_results(
        self,
        pipeline_result: Dict[str, Any]
    ) -> None:
        """
        Display pipeline results.
        """

        print("\n")
        print("=" * 60)
        print("CHAMPION PIPELINE")
        print("=" * 60)

        print(
            f"Model      : "
            f"{pipeline_result['model_name']}"
        )

        print(
            f"Model URI  : "
            f"{pipeline_result['model_uri']}"
        )

        print(
            f"Artifact   : "
            f"{pipeline_result['artifact_path']}"
        )
