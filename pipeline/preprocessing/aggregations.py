"""
aggregations.py

Forecasting aggregation layer.

Supports:

1. Enterprise Forecasting
2. Store Forecasting
3. Store + Department Forecasting
"""

from typing import Optional

import pandas as pd 

from pipeline.config.settings import (DATE_COLUMN,TARGET_COLUMN)


class WalmartAggregator:
    """
    Aggregation engine for Walmart forecasting.
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
        Standard forecasting format.

        Returns:
            Date
            Weekly_Sales
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
        Enterprise-level sales.

        Aggregate all stores
        and departments.
        """

        self._validate_dataframe(df)

        result = (
            df.groupby(DATE_COLUMN)[TARGET_COLUMN].sum().reset_index())

        return self._prepare_output(result)

    def get_store_sales(self,df: pd.DataFrame,store_id: int) -> pd.DataFrame:
        """
        Store-level sales.
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
        Store + Department sales.
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
        Top stores by total sales.

        Useful for:
        - experiment selection
        - reporting
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
        Top departments by sales.
        """

        self._validate_dataframe(df)

        return (
            df.groupby("Dept")[TARGET_COLUMN]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
    