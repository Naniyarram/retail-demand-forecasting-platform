"""
base_forecaster.py

Abstract forecasting contract for all forecasting models.


"""

from abc import ABC, abstractmethod 
from typing import Dict, Any

import pandas as pd

from pipeline.evaluation.metrics import ForecastMetrics


class BaseForecaster(ABC):
    """
    Unified forecasting interface.

    Enables:
    - Experiment Runner
    - Walk Forward Validation
    - MLflow Tracking
    - Model Registry
    - Champion Selection
    - Monitoring
    """

    def __init__(self):

        self.model = None

        self.model_name = self.__class__.__name__

        self.is_trained = False

        self.metrics = {}

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame
    ) -> None:
        """
        Train model.
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int
    ):
        """
        Generate forecast.
        """
        pass

    @abstractmethod
    def save_model(
        self,
        path: str
    ) -> None:
        """
        Save trained model.
        """
        pass

    @abstractmethod
    def load_model(
        self,
        path: str
    ) -> None:
        """
        Load model.
        """
        pass

    @abstractmethod
    def get_params(
        self
    ) -> Dict[str, Any]:
        """
        Return model hyperparameters.

        Required for:
        - MLflow logging
        - Reporting
        """
        pass

    def evaluate(
        self,
        y_true,
        y_pred
    ) -> Dict[str, float]:
        """
        Evaluate forecasts.
        """

        self.metrics = ForecastMetrics.evaluate(
            y_true,
            y_pred
        )

        return self.metrics

    def get_metrics(
        self
    ) -> Dict[str, float]:

        return self.metrics

    def get_model_name(
        self
    ) -> str:

        return self.model_name

    def get_model_info(
        self
    ) -> Dict[str, Any]:

        return {
            "model_name": self.model_name,
            "trained": self.is_trained,
            "parameters": self.get_params(),
            "metrics": self.metrics
        }