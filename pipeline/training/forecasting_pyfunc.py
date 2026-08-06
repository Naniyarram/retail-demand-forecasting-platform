"""
MLflow PyFunc wrapper to standardize the prediction interface for forecasting models.
"""

from typing import Any

import pandas as pd
import mlflow.pyfunc


class ForecastingPyFuncModel(
    mlflow.pyfunc.PythonModel
):
    """
    Wrapper for deploying models with a custom predict interface in MLflow.
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
        Predicts future values based on the 'horizon' parameter in the input dataframe.
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