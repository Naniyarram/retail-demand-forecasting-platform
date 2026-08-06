"""
Generates lags, rolling statistics, and calendar features for forecasting.
"""

import pandas as pd

from pipeline.config.settings import (
    DATE_COLUMN,
    TARGET_COLUMN
)


class FeatureEngineer:
    """
    Generates features from target time series data.
    """

    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        date_column: str = DATE_COLUMN
    ):

        self.target_column = target_column
        self.date_column = date_column

    def create_lag_features(
        self,
        df: pd.DataFrame,
        lags: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Generates historical lag values for the target column.
        """

        if lags is None:

            lags = [
                1,
                2,
                4,
                8,
                12,
                26,
                52
            ]

        df = df.copy()

        for lag in lags:

            df[f"lag_{lag}"] = (
                df[self.target_column]
                .shift(lag)
            )

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates rolling means and standard deviations to capture trends and volatility.
        """

        df = df.copy()

        windows = [4, 12, 26]

        for window in windows:

            df[f"rolling_mean_{window}"] = (
                df[self.target_column]
                .shift(1)
                .rolling(window)
                .mean()
            )

        std_windows = [4, 12]

        for window in std_windows:

            df[f"rolling_std_{window}"] = (
                df[self.target_column]
                .shift(1)
                .rolling(window)
                .std()
            )

        return df

    def create_calendar_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extracts temporal attributes like year, month, quarter, and week of year.
        """

        df = df.copy()

        df[self.date_column] = pd.to_datetime(
            df[self.date_column]
        )

        df["year"] = (
            df[self.date_column]
            .dt.year
        )

        df["month"] = (
            df[self.date_column]
            .dt.month
        )

        df["quarter"] = (
            df[self.date_column]
            .dt.quarter
        )

        df["week_of_year"] = (
            df[self.date_column]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        return df

    def create_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Runs the full feature engineering pipeline: sort, lag, rolling stats, calendar features, and drop NaNs.
        """

        df = (
            df
            .sort_values(
                self.date_column
            )
            .reset_index(
                drop=True
            )
        )

        df = self.create_lag_features(df)

        df = self.create_rolling_features(df)

        df = self.create_calendar_features(df)

        df = df.dropna()

        return df