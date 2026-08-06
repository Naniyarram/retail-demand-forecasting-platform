"""
Factory class to instantiate forecasting models by name.
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
    Registry and factory for all available forecasting models.
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
        Instantiates a forecaster model based on the provided name and keyword arguments.
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
        Lists the names of all registered forecaster classes.
        """

        return list(
            cls.SUPPORTED_MODELS.keys()
        )