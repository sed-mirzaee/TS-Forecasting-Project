# ============================================================
# Project entry point
# ============================================================

from src.config import STAGE, Stage, DATA_MODE, FORECAST_DATA_MODE
from src.config_training import TRAINING_CONFIG, SARIMAX_CANDIDATES
from src.config_forecasting import FORECASTING_CONFIG
from src.pipeline_data import run_data_preparation_pipeline
from src.pipeline_training import (autoarima_selection_pipeline,
                                   run_arima_training,
                                   run_training,)
from src.pipeline_forecasting import run_forecasting

# import warnings
# warnings.filterwarnings("ignore")

def main():

    # This stage is called two times: before training and before forecasting
    if STAGE == Stage.DATA_PREPARATION:
        run_data_preparation_pipeline(DATA_MODE, FORECAST_DATA_MODE)

    elif STAGE == Stage.FIND_ARIMA_ORDER:
        autoarima_selection_pipeline()

    elif STAGE == Stage.ARIMA_TRAINING:
        run_arima_training(SARIMAX_CANDIDATES)

    elif STAGE == Stage.TRAINING:
        run_training(TRAINING_CONFIG)

    elif STAGE == Stage.FORECASTING:
        run_forecasting(FORECASTING_CONFIG)

    else:
        raise ValueError(f"Unknown stage: {STAGE}")


if __name__ == "__main__":
    main()