"""
forecasting_pyfunc.py

Unified MLflow PyFunc wrapper for
forecasting models.

Responsibilities
----------------
- Wrap forecasting models
- Standardize prediction interface
- Enable MLflow model logging
- Support Model Registry

"""

from typing import Any

import pandas as pd
import mlflow.pyfunc


class ForecastingPyFuncModel(
    mlflow.pyfunc.PythonModel
):
    """
    Generic forecasting wrapper.

    Supports:
    - SARIMA
    - Prophet
    - XGBoost
    """

    def __init__(
        self,
        forecasting_model: Any
    ):
        self.forecasting_model = (
            forecasting_model
        )

    def predict(
        self,
        context,
        model_input: pd.DataFrame
    ):
        """
        MLflow prediction entrypoint.

        Expected Input
        --------------
        model_input

        Must contain:

            horizon

        Example
        -------
        horizon
        12

        Returns
        -------
        Forecast values
        """

        if "horizon" not in model_input.columns:

            raise ValueError(
                "model_input must contain "
                "'horizon' column."
            )

        horizon = int(
            model_input.iloc[0]["horizon"]
        )

        predictions = (
            self.forecasting_model.predict(
                horizon=horizon
            )
        )

        return pd.DataFrame(
            {
                "forecast":
                    predictions
            }
        )