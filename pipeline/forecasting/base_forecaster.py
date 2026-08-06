"""
Base interface for all forecasting models in the pipeline.
"""

from abc import ABC, abstractmethod 
from typing import Dict, Any

import pandas as pd

from pipeline.evaluation.metrics import ForecastMetrics


class BaseForecaster(ABC):
    """
    Abstract base class defining the shared interface for forecasting models.
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
        Fit the model to the training dataset.
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int
    ):
        """
        Generate forecasts for the specified horizon.
        """
        pass

    @abstractmethod
    def save_model(
        self,
        path: str
    ) -> None:
        """
        Save the trained model to disk.
        """
        pass

    @abstractmethod
    def load_model(
        self,
        path: str
    ) -> None:
        """
        Load a trained model from disk.
        """
        pass

    @abstractmethod
    def get_params(
        self
    ) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        """
        pass

    def evaluate(
        self,
        y_true,
        y_pred
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics comparing predictions against ground truth.
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