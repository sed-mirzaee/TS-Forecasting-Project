# =======================================================================================
# Data stages pipeline: loading, validating, preparation, processing, feature engineering
# =======================================================================================

import pandas as pd
from src.config import MATERIALS, Data_mode, Forecast_data_mode
from src.config import (INPUT_FILE,
                        MATERIAL_FILE,
                        PUBLIC_HOLIDAYS_FILE,
                        SCHOOL_HOLIDAYS_FILE,
                        TRAINING_DATA_FILE,
                        FORECASTING_DATA_FILE,
                        START_DATE,
                        FORECAST_FULL_INPUT_FILE,
                        FORECAST_FULL_MATERIAL_FILE,
                        FORECAST_APPEND_INPUT_FILE,
                        FORECAST_APPEND_MATERIAL_FILE,
)
from src.data_loader import (load_raw_data,
                             load_forecast_raw_data,
                             get_public_holidays,
                             load_school_holidays,
                             save_prepared_data, )
from src.preprocessing_common import (clean_input,
                                      clean_materials,
                                      keep_business_days,
                                      add_calendar_features, )

"""
Orchestrating data preparation
"""
def run_data_preparation_pipeline(data_mode: Data_mode, forecast_data_mode: Forecast_data_mode):

    # Load raw data
    if data_mode == Data_mode.TRAINING:
        df_input, df_materials = load_raw_data(INPUT_FILE, MATERIAL_FILE)

    elif data_mode == Data_mode.FORECASTING:
        if forecast_data_mode == Forecast_data_mode.FULL:
            df_input, df_materials = load_forecast_raw_data("FULL", INPUT_FILE, MATERIAL_FILE, FORECAST_FULL_INPUT_FILE, FORECAST_FULL_MATERIAL_FILE)
        else:
            df_input, df_materials = load_forecast_raw_data("APPEND", INPUT_FILE, MATERIAL_FILE, FORECAST_APPEND_INPUT_FILE, FORECAST_APPEND_MATERIAL_FILE)
    else:
        raise ValueError(
            f"Unknown data pipeline mode: {data_mode}"
        )

    # Clean raw data
    df_input = clean_input(df_input)
    df_materials = clean_materials(df_materials, MATERIALS)

    # Merge datasets
    df_base = merge_base_data(df_input, df_materials)

    # Keep business days
    df_base = keep_business_days(df_base)

    # Get public holidays
    df_holidays = get_public_holidays(PUBLIC_HOLIDAYS_FILE)

    # Load school holidays
    df_school_holidays = load_school_holidays(SCHOOL_HOLIDAYS_FILE)

    # Add external data
    df_base = add_external_data(df_base, df_holidays, df_school_holidays)

    # Add basic calendar features
    df_base = add_calendar_features(df_base)

    # set start of data
    df_base = df_base.loc[df_base["Date"] >= pd.Timestamp(START_DATE)].copy()
    df_base = df_base.reset_index(drop=True)

    # Remove unimportant features like holiday_name, ...

    # Save prepared data for next stage
    if data_mode == Data_mode.TRAINING:
        output_file = TRAINING_DATA_FILE

    elif data_mode == Data_mode.FORECASTING:
        output_file = FORECASTING_DATA_FILE

    else:
        raise ValueError(f"Unknown data mode: {data_mode}")

    save_prepared_data(df_base, output_file)

    return df_base

"""
Join Input and Output on Date
"""
def merge_base_data(df_input: pd.DataFrame, df_materials: pd.DataFrame, ) -> pd.DataFrame:

    df_merged = pd.merge(df_input, df_materials, on="Date", how="outer")
    df_merged = df_merged.sort_values("Date").reset_index(drop=True)

    return df_merged

"""
Join Base Data and External Data Sources on Date
"""
def add_external_data(df_base: pd.DataFrame, df_holidays: pd.DataFrame, df_school_holidays: pd.DataFrame,) -> pd.DataFrame:

    df = df_base.copy()

    df = df.merge(df_holidays, on="Date", how="left")

    df = df.merge(df_school_holidays, on="Date", how="left")

    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["is_school_holiday"] = df["is_school_holiday"].fillna(0).astype(int)

    return df