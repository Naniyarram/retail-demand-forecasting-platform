"""
XGBoost forecasting model wrapper.
"""

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from pipeline.config.settings import (
    TARGET_COLUMN,
    DATE_COLUMN,
    RANDOM_STATE
)

from pipeline.forecasting.base_forecaster import (
    BaseForecaster
)

from pipeline.preprocessing.feature_engineering import (
    FeatureEngineer
)


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost autoregressive/recursive forecaster.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ):
        super().__init__()

        self.model_name = "XGBoost"

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.feature_engineer = FeatureEngineer()

        self.feature_columns = None

        self.training_data = None

        self.fitted_model = None

    def fit(
        self,
        train_df: pd.DataFrame
    ) -> None:
        """
        Fit the XGBoost regressor using engineered features.
        """

        featured_df = (
            self.feature_engineer
            .create_features(train_df)
        )

        X = featured_df.drop(
            columns=[
                TARGET_COLUMN,
                DATE_COLUMN
            ]
        )

        y = featured_df[TARGET_COLUMN]

        self.feature_columns = X.columns.tolist()

        self.training_data = train_df.copy()

        self.fitted_model = XGBRegressor(
            objective="reg:squarederror",

            n_estimators=self.n_estimators,

            learning_rate=self.learning_rate,

            max_depth=self.max_depth,

            subsample=self.subsample,

            colsample_bytree=self.colsample_bytree,

            random_state=RANDOM_STATE
        )

        self.fitted_model.fit(X, y)

        self.model = self.fitted_model

        self.is_trained = True

    def predict(
        self,
        horizon: int
    ) -> np.ndarray:
        """
        Generate recursive multi-step forecasts by appending predictions as lags.
        """

        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained."
            )

        history = (
            self.training_data.copy()
            .reset_index(drop=True)
        )

        predictions = []

        for _ in range(horizon):

            featured_df = (
                self.feature_engineer
                .create_features(history)
            )

            latest_row = (
                featured_df
                .iloc[[-1]]
                .drop(
                    columns=[
                        TARGET_COLUMN,
                        DATE_COLUMN
                    ]
                )
            )

            pred = float(
                self.fitted_model.predict(
                    latest_row
                )[0]
            )

            predictions.append(pred)

            next_date = (
                history[DATE_COLUMN]
                .max()
                + pd.Timedelta(
                    weeks=1
                )
            )

            new_row = pd.DataFrame(
                {
                    DATE_COLUMN: [next_date],
                    TARGET_COLUMN: [pred]
                }
            )

            history = pd.concat(
                [
                    history,
                    new_row
                ],
                ignore_index=True
            )

        return np.array(predictions)

    def save_model(
        self,
        path: str
    ) -> None:
        """
        Save model and metadata artifact package.
        """

        if not self.is_trained:
            raise RuntimeError(
                "No trained model found."
            )

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        model_package = {
            "model": self.fitted_model,
            "feature_columns": self.feature_columns,
            "training_data": self.training_data,
            "params": self.get_params()
        }

        joblib.dump(
            model_package,
            path
        )

    def load_model(
        self,
        path: str
    ) -> None:
        """
        Load model and state package.
        """

        package = joblib.load(path)

        self.fitted_model = package["model"]

        self.feature_columns = (
            package["feature_columns"]
        )

        self.training_data = (
            package["training_data"]
        )

        self.model = self.fitted_model

        self.is_trained = True

    def get_params(
        self
    ) -> Dict[str, Any]:
        """
        Get model hyperparameters.
        """

        return {

            "model_type": "XGBoost",

            "n_estimators":
                self.n_estimators,

            "learning_rate":
                self.learning_rate,

            "max_depth":
                self.max_depth,

            "subsample":
                self.subsample,

            "colsample_bytree":
                self.colsample_bytree
        }