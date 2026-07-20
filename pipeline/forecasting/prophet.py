"""
prophet.py

Prophet forecasting model.

Supports:
- Walk-forward validation
- Experiment Runner
- MLflow integration
- Model persistence


"""

from pathlib import Path
from typing import Dict, Any

import pandas as pd
import joblib

from prophet import Prophet

from pipeline.config.settings import (
    TARGET_COLUMN,
    DATE_COLUMN
)

from pipeline.forecasting.base_forecaster import (
    BaseForecaster
)


class ProphetForecaster(BaseForecaster):
    """
    Prophet forecasting model.

    Converts:

    Date -> ds
    Weekly_Sales -> y

    internally.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "multiplicative"
    ):
        super().__init__()

        self.model_name = "Prophet"

        self.yearly_seasonality = yearly_seasonality

        self.seasonality_mode = seasonality_mode

        self.fitted_model = None

    def _prepare_prophet_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert project schema
        into Prophet schema.
        """

        return (
            df[
                [
                    DATE_COLUMN,
                    TARGET_COLUMN
                ]
            ]
            .rename(
                columns={
                    DATE_COLUMN: "ds",
                    TARGET_COLUMN: "y"
                }
            )
            .copy()
        )

    def fit(
        self,
        train_df: pd.DataFrame
    ) -> None:
        """
        Train Prophet model.
        """

        prophet_df = self._prepare_prophet_dataframe(
            train_df
        )

        self.fitted_model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=self.seasonality_mode
        )

        self.fitted_model.fit(
            prophet_df
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

        future = self.fitted_model.make_future_dataframe(
            periods=horizon,
            freq="W"
        )

        forecast = self.fitted_model.predict(
            future
        )

        predictions = (
            forecast["yhat"]
            .tail(horizon)
            .values
        )

        return predictions

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
        Load trained model.
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
        Return model metadata.

        Used by:
        - MLflow
        - Reporting
        - Leaderboards
        """

        return {
            "model_type": "Prophet",
            "yearly_seasonality": self.yearly_seasonality,
            "seasonality_mode": self.seasonality_mode
        }