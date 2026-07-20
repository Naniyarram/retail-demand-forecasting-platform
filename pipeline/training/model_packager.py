"""
model_packager.py

Model packaging layer.

Responsibilities
----------------
- Serialize trained models
- Store model artifacts
- Return artifact paths


"""

from pathlib import Path
from typing import Any

import joblib

from pipeline.utils.helpers import save_json


class ModelPackager:
    """
    Packages trained forecasting models.
    """

    def __init__(
        self,
        artifact_dir: str = "artifacts/models"
    ):

        self.artifact_dir = Path(
            artifact_dir
        )

        self.artifact_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def package_model(
        self,
        model_name: str,
        model_object: Any
    ) -> str:
        """
        Serialize model artifact.

        Returns
        -------
        str
            Artifact path
        """

        model_path = (
            self.artifact_dir
            /
            f"{model_name.lower()}.pkl"
        )

        joblib.dump(
            model_object,
            model_path
        )

        return str(model_path)

    def package_champion(
        self,
        model_name: str,
        model_object: Any,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, str]:
        """
        Persist the production champion artifact and metadata.
        """

        champion_model_path = (
            self.artifact_dir
            / "champion_model.pkl"
        )

        champion_metadata_path = (
            self.artifact_dir
            / "champion_metadata.json"
        )

        champion_model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model_object,
            champion_model_path
        )

        metadata_payload = {
            "model_name": model_name,
            "model_type": type(model_object).__name__
        }

        if metadata:

            metadata_payload.update(
                metadata
            )

        metadata_path = save_json(
            metadata_payload,
            champion_metadata_path
        )

        return {
            "artifact_path": str(champion_model_path),
            "metadata_path": metadata_path
        }

    def artifact_exists(
        self,
        artifact_path: str
    ) -> bool:
        """
        Verify artifact exists.
        """

        return Path(
            artifact_path
        ).exists()

    def load_artifact(
        self,
        artifact_path: str
    ):
        """
        Load packaged model.
        """

        return joblib.load(
            artifact_path
        )
