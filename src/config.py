# ============================================================
# Project-wide configuration
# ============================================================

from enum import Enum
from pathlib import Path

# Project Config
#----------------------------------------

# Project Stages for controlling pipelines
class Stage(Enum):
    DATA_PREPARATION = "data_preparation"
    TRAINING = "training"
    FIND_ARIMA_ORDER = "find_arima_order"
    ARIMA_TRAINING = "arima_training"
    FORECASTING = "forecasting"

STAGE = Stage.FORECASTING

class Data_mode(Enum):
    TRAINING = "training"
    FORECASTING = "forecasting"

DATA_MODE = Data_mode.FORECASTING

class Forecast_data_mode(Enum):
    FULL = "full"
    APPEND = "append"

FORECAST_DATA_MODE = Forecast_data_mode.FULL
# "FULL"   → use forecast raw files as complete datasets
# "APPEND" → append forecast raw files to historical raw data

# Main Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

INPUT_FILE = RAW_DATA_DIR / "eingangsbandwaage_studenten.csv"
MATERIAL_FILE = RAW_DATA_DIR / "ballengewichte_studenten.csv"
SCHOOL_HOLIDAYS_FILE = RAW_DATA_DIR / "school_holidays.csv"
PUBLIC_HOLIDAYS_FILE = PROCESSED_DATA_DIR / "public_holidays.csv"

FORECAST_FULL_INPUT_FILE = RAW_DATA_DIR / "eingangsbandwaage_studenten_all.csv"
FORECAST_FULL_MATERIAL_FILE = RAW_DATA_DIR / "ballengewichte_studenten_all.csv"

FORECAST_APPEND_INPUT_FILE = RAW_DATA_DIR / "eingangsbandwaage_studenten_2025.csv"
FORECAST_APPEND_MATERIAL_FILE = RAW_DATA_DIR / "ballengewichte_studenten_2025.csv"

TRAINING_DATA_FILE = PROCESSED_DATA_DIR / "prepared_training.csv"
FORECASTING_DATA_FILE = PROCESSED_DATA_DIR / "prepared_forecasting.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training"
FORECASTING_OUTPUT_DIR = OUTPUT_DIR / "forecasts"

# Competition targets and horizons (business days).
TIME_FREQUENCY = "B"
WEEK_DAYS = 5 # Business Day
MATERIALS = [str(i) for i in range(1, 14)]  # 1, 2, .., 13
HORIZONS = [10,  # Two weeks without weekends
            22,  # One month
            66,  # Three months
            132] # Six months

# The fraction of training set
INITIAL_RATIO = 0.789 # ~ 80% of data but keep complete folds for montly horizons

# Start Date for training
START_DATE = "2020-03-23"

# Evaluation
SMAPE_EPS = 1 / 1000

# Data processing
#----------------------------------------
ZERO_BLOCK_THRESHOLD = 5

