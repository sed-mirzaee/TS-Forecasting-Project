# ============================================================
# Data processing methods
# ============================================================

import numpy as np
import pandas as pd
from src.config import ZERO_BLOCK_THRESHOLD

"""
Clean invalid Input observations while preserving timestamps.

Rules retrieved from 01_data_quality_eda
----------------------------------------
1. Negative Input values are replaced with NaN.
2. Continuous zero blocks longer than the configured threshold are replaced with NaN.
3. Short zero blocks are kept unchanged.
"""
def clean_input(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    # Validate raw columns
    required_cols = {"Zeitstempel", "Masse"}
    missing_cols = required_cols - set(df_clean.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required Input columns: {sorted(missing_cols)}"
        )

    # Standardize column names, Prepare Date
    df_clean = df_clean.rename(
        columns={"Zeitstempel": "Date"}
    )

    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean = df_clean.sort_values("Date").reset_index(drop=True)

    # Check duplicate dates
    duplicate_dates = df_clean["Date"].duplicated().sum()

    if duplicate_dates > 0:
        raise ValueError(
            f"Material data contains {duplicate_dates} duplicate dates."
        )

    # Align Date(Day)
    df_clean = align_daily_calendar(df_clean)

    # Negative Input values -> NaN
    df_clean.loc[df_clean["Masse"] < 0, "Masse"] = np.nan

    # Detect continuous zero blocks
    is_zero = df_clean["Masse"].eq(0)
    block_id = is_zero.ne(is_zero.shift()).cumsum()
    zero_block_length = (is_zero.groupby(block_id).transform("sum"))

    # Long zero blocks -> NaN
    long_zero_block = (is_zero & (zero_block_length > ZERO_BLOCK_THRESHOLD))
    df_clean.loc[long_zero_block, "Masse"] = np.nan

    return df_clean

"""
Standardize, clean and validate observed material-fraction data.

Rules retrieved from 01_data_quality_eda
----------------------------------------
1. All 13 material columns must exist.
2. Observed rows must contain all 13 fractions.
3. Each fraction must lie within [0, 1].
4. The 13 fractions must sum to 1 within numerical tolerance.
5. Zero fractions are valid values.
6. Missing days observations are explicitly represented as `NA`

No automatic normalization or imputation is performed.
"""
def clean_materials(df: pd.DataFrame, material_cols: list[str],) -> pd.DataFrame:
    df_clean = df.copy()

    # Validate required raw columns
    required_cols = {"Datum", *material_cols}

    missing_cols = required_cols - set(df_clean.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required Material columns: {sorted(missing_cols)}"
        )

    #  Standardize column names, Prepare Date
    df_clean = df_clean.rename(columns={"Datum": "Date"})
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean = (df_clean.sort_values("Date").reset_index(drop=True))

    # Check duplicate dates
    duplicate_dates = df_clean["Date"].duplicated().sum()

    if duplicate_dates > 0:
        raise ValueError(
            f"Material data contains {duplicate_dates} duplicate dates."
        )

    # Validate missing material values
    if df_clean[material_cols].isna().any().any():
        raise ValueError("Missing values found in observed Material rows.")

    # Validate fraction range
    invalid_range = ((df_clean[material_cols] < 0) | (df_clean[material_cols] > 1)).any(axis=1)

    if invalid_range.any():
        raise ValueError(
            f"{invalid_range.sum()} Material rows contain values outside [0, 1]."
        )

    # Validate fraction sum
    material_sum = df_clean[material_cols].sum(axis=1)

    invalid_sum = (material_sum - 1).abs() > 1e-10

    if invalid_sum.any():
        raise ValueError(
            f"{invalid_sum.sum()} Material rows do not sum to 1."
        )

    # Align Date (Day)
    df_clean = align_daily_calendar(df_clean)

    return df_clean

"""
Align a dataset to a complete daily calendar.
"""
def align_daily_calendar(df: pd.DataFrame, date_col: str = "Date", ) -> pd.DataFrame:

    df_aligned = (
        df
        .copy()
        .sort_values(date_col)
        .set_index(date_col)
        .asfreq("D")
        .reset_index()
    )

    return df_aligned

"""
Align a dataset to a complete daily calendar.
"""
def keep_business_days(df: pd.DataFrame, date_col: str = "Date", ) -> pd.DataFrame:

    return (
        df[df[date_col].dt.dayofweek < 5]
        .copy()
        .reset_index(drop=True)
    )

"""
Add calendar features.
"""
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = df.copy()

    df_features["day_of_week"] = df_features["Date"].dt.dayofweek
    df_features["month"] = df_features["Date"].dt.month
    df_features["year"] = df_features["Date"].dt.year

    return df_features