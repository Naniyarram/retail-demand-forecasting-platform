from pathlib import Path


# Project Paths


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

MODEL_DIR = ARTIFACTS_DIR / "models"

REPORT_DIR = ARTIFACTS_DIR / "reports"

LEADERBOARD_DIR = ARTIFACTS_DIR / "leaderboards"

CHAMPION_MODEL_PATH = MODEL_DIR / "champion_model.pkl"

CHAMPION_METADATA_PATH = MODEL_DIR / "champion_metadata.json"

DRIFT_REPORT_DIR = REPORT_DIR / "drift"


# Dataset


TARGET_COLUMN = "Weekly_Sales"

DATE_COLUMN = "Date"


# Forecasting

FORECAST_HORIZON = 12

MAX_FORECAST_HORIZON = 52

N_SPLITS = 5



# Experiment


EXPERIMENT_NAME = "Retail Demand Forecasting"

PRIMARY_METRIC = "MAPE"


# Randomness


RANDOM_STATE = 42
