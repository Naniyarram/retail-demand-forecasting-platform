"""
Aggregates sales data at different levels (company, store, department)
for forecasting.
"""

from typing import Optional

import pandas as pd 

from pipeline.config.settings import (DATE_COLUMN,TARGET_COLUMN)


class WalmartAggregator:
    """
    Helper class to handle data aggregation at different hierarchy levels.
    """

    @staticmethod
    def _validate_dataframe( df: pd.DataFrame) -> None:

        required_columns = {DATE_COLUMN, TARGET_COLUMN,"Store","Dept"}

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

    @staticmethod
    def _prepare_output(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures the data is sorted by date and contains only the required columns.
        """

        df = (df.sort_values(DATE_COLUMN).reset_index(drop=True))

        return df[
            [
                DATE_COLUMN,
                TARGET_COLUMN
            ]
        ]

    def get_company_sales(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates sales across all stores and departments to get company-wide totals.
        """

        self._validate_dataframe(df)

        result = (
            df.groupby(DATE_COLUMN)[TARGET_COLUMN].sum().reset_index())

        return self._prepare_output(result)

    def get_store_sales(self,df: pd.DataFrame,store_id: int) -> pd.DataFrame:
        """
        Aggregates sales for a specific store.
        """

        self._validate_dataframe(df)

        result = (
            df[df["Store"] == store_id].groupby(DATE_COLUMN)[TARGET_COLUMN].sum().reset_index())

        if result.empty:
            raise ValueError(
                f"Store {store_id} not found."
            )

        return self._prepare_output(result)

    def get_store_department_sales( self, df: pd.DataFrame,store_id: int,dept_id: int) -> pd.DataFrame:
        """
        Aggregates sales for a specific department within a store.
        """

        self._validate_dataframe(df)

        result = (
            df[
                (df["Store"] == store_id)
                &
                (df["Dept"] == dept_id)
            ]
            .groupby(DATE_COLUMN)[TARGET_COLUMN]
            .sum()
            .reset_index()
        )

        if result.empty:
            raise ValueError(
                f"No data found for "
                f"Store={store_id}, "
                f"Dept={dept_id}"
            )

        return self._prepare_output(result)

    def get_top_stores(self,df: pd.DataFrame,top_n: int = 10) -> pd.DataFrame:
        """
        Identifies the highest-performing stores by overall sales volume.
        """

        self._validate_dataframe(df)

        return (
            df.groupby("Store")[TARGET_COLUMN]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

    def get_top_departments(self,df: pd.DataFrame,top_n: int = 10) -> pd.DataFrame:
        """
        Identifies the highest-performing departments by overall sales volume.
        """

        self._validate_dataframe(df)

        return (
            df.groupby("Dept")[TARGET_COLUMN]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
    