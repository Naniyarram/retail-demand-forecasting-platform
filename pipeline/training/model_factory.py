"""
model_factory.py

Centralized forecasting model factory.

Responsibilities
----------------
- Create forecasting models
- Centralize model initialization
- Support retraining workflows
- Support deployment workflows

"""

from pipeline.forecasting.sarima import (
    SARIMAForecaster
)

from pipeline.forecasting.prophet import (
    ProphetForecaster
)

from pipeline.forecasting.xgboost import (
    XGBoostForecaster
)


class ModelFactory:
    """
    Factory for forecasting models.
    """

    SUPPORTED_MODELS = {
        "SARIMA": SARIMAForecaster,
        "Prophet": ProphetForecaster,
        "XGBoost": XGBoostForecaster
    }

    @classmethod
    def create_model(
        cls,
        model_name: str,
        **kwargs
    ):
        """
        Create forecasting model.

        Parameters
        ----------
        model_name : str

        Returns
        -------
        Forecasting Model
        """

        if model_name not in cls.SUPPORTED_MODELS:

            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        model_class = (
            cls.SUPPORTED_MODELS[
                model_name
            ]
        )

        return model_class(
            **kwargs
        )

    @classmethod
    def list_models(
        cls
    ):
        """
        Return supported models.
        """

        return list(
            cls.SUPPORTED_MODELS.keys()
        )