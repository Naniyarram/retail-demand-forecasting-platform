"""
sarima.py

Auto-SARIMA forecasting model.

Supports:
- Walk-forward validation
- Experiment Runner
- MLflow integration
- Model persistence


"""

from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from pmdarima import auto_arima

from pipeline.config.settings import (
    TARGET_COLUMN,
    DATE_COLUMN
)

from pipeline.forecasting.base_forecaster import (
    BaseForecaster
)


class SARIMAForecaster(BaseForecaster):

    def __init__(
        self,
        seasonal_period: int = 52
    ):

        super().__init__()

        self.model_name = "SARIMA"

        self.seasonal_period = seasonal_period

        self.fitted_model = None

    def fit(
        self,
        train_df: pd.DataFrame
    ) -> None:
        """
        Train Auto-SARIMA model.
        """

        if TARGET_COLUMN not in train_df.columns:
            raise ValueError(
                f"{TARGET_COLUMN} not found."
            )

        train_series = (
            train_df
            .sort_values(DATE_COLUMN)
            [TARGET_COLUMN]
        )

        self.fitted_model = auto_arima(
            train_series,

            seasonal=True,
            m=self.seasonal_period,

            start_p=0,
            start_q=0,

            max_p=5,
            max_q=5,

            start_P=0,
            start_Q=0,

            max_P=2,
            max_Q=2,

            d=None,
            D=None,

            trace=False,

            error_action="ignore",

            suppress_warnings=True,

            stepwise=True,

            information_criterion="aic"
        )

        self.model = self.fitted_model

        self.is_trained = True

    def predict(
        self,
        horizon: int
    ):
        """
        Forecast future periods.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

        forecasts = self.fitted_model.predict(
            n_periods=horizon
        )

        return forecasts

    def save_model(
        self,
        path: str
    ) -> None:
        """
        Save trained model.
        """

        if not self.is_trained:
            raise RuntimeError(
                "No trained model found."
            )

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            self.fitted_model,
            path
        )

    def load_model(
        self,
        path: str
    ) -> None:
        """
        Load saved model.
        """

        self.fitted_model = joblib.load(
            path
        )

        self.model = self.fitted_model

        self.is_trained = True

    def get_params(
        self
    ) -> Dict[str, Any]:
        """
        Model metadata.

        Useful for:
        - MLflow
        - Reporting
        - Leaderboards
        """

        if not self.is_trained:

            return {
                "seasonal_period": self.seasonal_period
            }

        order = self.fitted_model.order
        seasonal_order = self.fitted_model.seasonal_order

        return {

            "model_type": "SARIMA",

            "seasonal_period": self.seasonal_period,

            "p": order[0],
            "d": order[1],
            "q": order[2],

            "P": seasonal_order[0],
            "D": seasonal_order[1],
            "Q": seasonal_order[2],
            "m": seasonal_order[3]
        }