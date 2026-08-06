"""
Walk-forward validation for time series forecasting models.
"""

from typing import Generator
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from pipeline.config.settings import (
    FORECAST_HORIZON,
    N_SPLITS
)


class WalkForwardValidator:
    """
    Split time series data using rolling/expanding windows to prevent lookahead bias.
    """

    def __init__(
        self,
        n_splits: int = N_SPLITS,
        forecast_horizon: int = FORECAST_HORIZON
    ):
        self.n_splits = n_splits
        self.forecast_horizon = forecast_horizon

    def split(
        self,
        df: pd.DataFrame
    ) -> Generator[
        Tuple[np.ndarray, np.ndarray],
        None,
        None
    ]:
        """
        Generates walk-forward training and test split indices.
        """

        if len(df) == 0:
            raise ValueError(
                "Input dataframe is empty."
            )

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.forecast_horizon
        )

        for train_idx, test_idx in tscv.split(df):

            yield train_idx, test_idx

    def get_fold_summary(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Returns metadata for each validation split as a DataFrame.
        """

        folds = []

        for fold_number, (
            train_idx,
            test_idx
        ) in enumerate(
            self.split(df),
            start=1
        ):

            folds.append(
                {
                    "fold": fold_number,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "train_start": train_idx[0],
                    "train_end": train_idx[-1],
                    "test_start": test_idx[0],
                    "test_end": test_idx[-1]
                }
            )

        return pd.DataFrame(folds)