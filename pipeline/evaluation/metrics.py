"""
metrics.py

Centralized evaluation metrics for forecasting models.

Used by:
- SARIMA
- Prophet
- XGBoost
- Experiment Runner
- MLflow Tracking
- Model Monitoring


"""

from typing import Dict

import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,)


class ForecastMetrics:
    """
    Standard forecasting evaluation metrics.
    """

    @staticmethod
    def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate prediction inputs.

        Returns cleaned numpy arrays.
        """

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_true) == 0:
            raise ValueError(
                "y_true is empty."
            )

        if len(y_pred) == 0:
            raise ValueError(
                "y_pred is empty."
            )

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Shape mismatch: "
                f"{len(y_true)} vs {len(y_pred)}"
            )

        mask = (
            np.isfinite(y_true)
            & np.isfinite(y_pred)
        )

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            raise ValueError(
                "No valid observations after cleaning."
            )

        return y_true, y_pred

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""

        y_true, y_pred = ForecastMetrics._validate_inputs(y_true,y_pred)

        return float(
            np.sqrt(
                mean_squared_error(y_true,y_pred)
            )
        )

    @staticmethod
    def mae(y_true: np.ndarray,y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""

        y_true, y_pred = ForecastMetrics._validate_inputs(y_true,y_pred)

        return float(mean_absolute_error(y_true,y_pred))

    @staticmethod
    def mape(y_true: np.ndarray,y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error.

        Handles zero actual values safely.
        """

        y_true, y_pred = ForecastMetrics._validate_inputs(y_true,y_pred )

        mask = y_true != 0

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return np.nan

        mape = np.mean( np.abs((y_true - y_pred)/ y_true )) * 100
        return float(mape)

    @staticmethod
    def evaluate(y_true: np.ndarray,y_pred: np.ndarray) -> Dict[str, float]:
        """
        Return all forecasting metrics.
        """

        return {
            "RMSE": ForecastMetrics.rmse( y_true, y_pred ),
            "MAE": ForecastMetrics.mae(y_true,y_pred),
            "MAPE": ForecastMetrics.mape( y_true, y_pred)
        }