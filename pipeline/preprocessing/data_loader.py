"""
Loads and validates the Walmart forecasting datasets.
"""

from pathlib import Path

import pandas as pd

from pipeline.config.settings import DATA_DIR


class WalmartDataLoader:
    """
    Data loader and validator for the retail forecasting datasets.
    """

    REQUIRED_TRAIN_COLUMNS = {
        "Store",
        "Dept",
        "Date",
        "Weekly_Sales",
        "IsHoliday"
    }

    REQUIRED_FEATURE_COLUMNS = {
        "Store",
        "Date",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "IsHoliday"
    }

    REQUIRED_STORE_COLUMNS = {
        "Store",
        "Type",
        "Size"
    }

    def __init__(self):
        self.train_path = DATA_DIR / "train.csv"
        self.features_path = DATA_DIR / "features.csv"
        self.stores_path = DATA_DIR / "stores.csv"
        self.test_path = DATA_DIR / "test.csv"

    def _validate_file_exists(
        self,
        file_path: Path
    ) -> None:
        """
        Checks if the file exists at the given path.
        """

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

    def _parse_dates(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Parses dates to datetime format.
        """

        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"]
            )

        return df

    def _validate_columns(
        self,
        df: pd.DataFrame,
        required_columns: set,
        dataset_name: str
    ) -> None:
        """
        Verifies that all required columns are present in the dataframe.
        """

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{dataset_name} missing columns: "
                f"{missing}"
            )

    def _remove_duplicates(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Drops duplicate records.
        """

        return df.drop_duplicates()

    def load_train_data(
        self
    ) -> pd.DataFrame:

        self._validate_file_exists(
            self.train_path
        )

        df = pd.read_csv(
            self.train_path
        )

        self._validate_columns(
            df,
            self.REQUIRED_TRAIN_COLUMNS,
            "train.csv"
        )

        df = self._parse_dates(df)

        df = self._remove_duplicates(df)

        return df

    def load_features_data(
        self
    ) -> pd.DataFrame:

        self._validate_file_exists(
            self.features_path
        )

        df = pd.read_csv(
            self.features_path
        )

        self._validate_columns(
            df,
            self.REQUIRED_FEATURE_COLUMNS,
            "features.csv"
        )

        df = self._parse_dates(df)

        df = self._remove_duplicates(df)

        return df

    def load_stores_data(
        self
    ) -> pd.DataFrame:

        self._validate_file_exists(
            self.stores_path
        )

        df = pd.read_csv(
            self.stores_path
        )

        self._validate_columns(
            df,
            self.REQUIRED_STORE_COLUMNS,
            "stores.csv"
        )

        df = self._remove_duplicates(df)

        return df

    def load_test_data(
        self
    ) -> pd.DataFrame:

        self._validate_file_exists(
            self.test_path
        )

        df = pd.read_csv(
            self.test_path
        )

        df = self._parse_dates(df)

        df = self._remove_duplicates(df)

        return df

    def load_all(
        self
    ) -> dict:
        """
        Loads train, features, stores, and test datasets in one go.
        """

        return {
            "train": self.load_train_data(),
            "features": self.load_features_data(),
            "stores": self.load_stores_data(),
            "test": self.load_test_data()
        }