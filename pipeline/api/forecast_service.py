"""
forecast_service.py

Application service for loading the champion model artifact
and generating forecasts.


"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pipeline.config.settings import (
    CHAMPION_MODEL_PATH,
    CHAMPION_METADATA_PATH,
    MAX_FORECAST_HORIZON
)
from pipeline.utils.helpers import load_json


class ForecastService:
    """
    Serves forecasts from a packaged champion model.
    """

    def __init__(
        self,
        artifact_path: str | Path = CHAMPION_MODEL_PATH,
        metadata_path: str | Path = CHAMPION_METADATA_PATH
    ):

        self.artifact_path = Path(artifact_path)

        self.metadata_path = Path(metadata_path)

        self.model = None

        self.metadata: dict[str, Any] = {}

    def load_model(self) -> None:
        """
        Load the champion artifact into memory.
        """

        if not self.artifact_path.exists():

            raise FileNotFoundError(
                "Champion model artifact not found: "
                f"{self.artifact_path}. Run the champion "
                "pipeline before starting the API."
            )

        self.model = joblib.load(
            self.artifact_path
        )

        if self.metadata_path.exists():

            self.metadata = load_json(
                self.metadata_path
            )

    def is_model_loaded(self) -> bool:
        """
        Return whether a model artifact is ready.
        """

        return self.model is not None

    def get_model_name(self) -> str:
        """
        Resolve a human-readable model name.
        """

        if self.metadata.get("model_name"):

            return str(
                self.metadata["model_name"]
            )

        if hasattr(self.model, "get_model_name"):

            return str(
                self.model.get_model_name()
            )

        if self.model is not None:

            return type(self.model).__name__

        return "unknown"

    def get_metadata(self) -> dict[str, Any]:
        """
        Return model serving metadata.
        """

        return {
            "model_name": self.get_model_name(),
            "artifact_path": str(self.artifact_path),
            "metadata": self.metadata
        }

    def forecast(
        self,
        forecast_horizon: int
    ) -> list[float]:
        """
        Generate a forecast from the loaded champion.
        """

        if not self.is_model_loaded():

            self.load_model()

        if forecast_horizon < 1:

            raise ValueError(
                "forecast_horizon must be at least 1."
            )

        if forecast_horizon > MAX_FORECAST_HORIZON:

            raise ValueError(
                "forecast_horizon must be less than or "
                f"equal to {MAX_FORECAST_HORIZON}."
            )

        if not hasattr(self.model, "predict"):

            raise TypeError(
                "Loaded artifact does not implement predict(horizon)."
            )

        predictions = self.model.predict(
            horizon=forecast_horizon
        )

        return (
            np.asarray(predictions, dtype=float)
            .round(2)
            .tolist()
        )
