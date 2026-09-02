# ============================================================
# Forecasting models
# ============================================================

import numpy as np
import pandas as pd

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.preprocessing_model_based import (arima_imputation, ridge_history_series)
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge

def forecast_naive(
    train: pd.DataFrame,
    material: str,
    horizon: int,
) -> np.ndarray:

    y_train = train[material].dropna()

    # if there is nothinng to train
    if y_train.empty:
        return np.full(horizon, np.nan)

    last_value = y_train.iloc[-1]

    return np.full(horizon, last_value)

def forecast_moving_average(
    train: pd.DataFrame,
    material: str,
    horizon: int,
    window: int | None = None, # I assumed window = horizon
) -> np.ndarray:

    if window is None:
        window = horizon

    # the history is exactly as window (missings remove when average is computed)
    history = (train[material].astype(float).iloc[-window:].tolist())

    # if there is no info. It doesn't happen because first missing GAP appeare in 2021, middle of training data part
    if np.isnan(history).all():
        return np.full(horizon, np.nan)

    predictions = []

    for _ in range(horizon):

        # 1. select a window size block
        # 2. remove missings -> this help to keep time periods
        # 3. compute Average without missing values
        next_prediction = float(np.nanmean(history[-window:]))

        predictions.append(next_prediction)

        # move window forward with predicted value
        history.append(next_prediction)

    return np.asarray(predictions, dtype=float)

# Fit ARIMA/SARIMA without dropping or imputing missing rows. For arima, seasonal orders are empty.
def forecast_sarimax(
    train: pd.DataFrame,
    material: str,
    horizon: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> np.ndarray:

    y_train = (train[material].astype(float).copy())

    if y_train.notna().sum() == 0:
        raise ValueError(
            f"No observed training values for "
            f"Material {material}."
        )

    model = SARIMAX(
        endog=y_train,
        order=order,
        seasonal_order=seasonal_order,

        missing="none", # To fit model with existing missing values

        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        fitted_model = model.fit(disp=False, maxiter=500,)

    prediction = (fitted_model.forecast(steps=horizon).to_numpy(dtype=float))

    return prediction

# Ridge forecasting, Linear Regression with alpha penalty for regularization
def forecast_ridge(
    train: pd.DataFrame,
    material: str,
    horizon: int,
    lags: list[int],
    future: pd.DataFrame | None = None,
    alpha: float = 1.0,
    rolling_windows: list[int] | None = None,
    use_day_of_week: bool = False,
    use_holiday: bool = False,
    use_school_holiday: bool = False,
    input_lags: list[int] | None = None,
    input_rolling: list[tuple[int, int]] | None = None,
    history_fill_limit: int = 5,
) -> np.ndarray:
    """
    Missing-value handeling:
    1. y_raw is kept unchanged and supplies the training target.
       Therefore, an originally missing target is never learned as if it
       were an observed value.
    2. history_series is forward-filled only for short gaps and is used
       only to construct lag features.
    3. Rows whose target or required lag features are still missing are
       removed before fitting.
    4. Optional Input/Masse features use only lagged pre-origin Input values
       (single lags or lagged rolling means); future/test Masse is never read.
       Every Input lag must be >= horizon.
    5. Future-known calendar features (e.g. day_of_week, is_holiday,
       is_school_holiday) are read directly from the prepared future/test block.
    6. During recursive forecasting, the fitted model is reused and each
       prediction is appended to history for the next step.
    """

    if not lags or any(lag <= 0 for lag in lags):
        raise ValueError("Ridge lags must be positive integers.")

    if rolling_windows is None:
        rolling_windows = []

    if input_lags is None:
        input_lags = []

    if input_rolling is None:
        input_rolling = []

    if any(window <= 0 for window in rolling_windows):
        raise ValueError("Ridge rolling windows must be positive integers.")

    if any(lag <= 0 for lag in input_lags):
        raise ValueError("Ridge Input lags must be positive integers.")

    for lag, window in input_rolling:
        if lag <= 0 or window <= 0:
            raise ValueError(
                "Ridge Input rolling specs must contain positive (lag, window) values."
            )

    # Future Input/Masse is not known at forecast time. To guarantee that
    # every Input lag used anywhere in a forecast block refers only to
    # observations available at the forecast origin, each Input lag must
    # be at least as large as the block horizon. Otherwise, there is data leakage.

    if any(lag < horizon for lag in input_lags):
        raise ValueError(
            "Ridge Input lags must be >= horizon to avoid using future Input values."
        )

    if any(lag < horizon for lag, _ in input_rolling):
        raise ValueError(
            "Ridge Input rolling lags must be >= horizon to avoid using future Input values."
        )

    # Original observed series: this is the target and is never imputed.
    y_raw = train[material].astype(float).copy()

    # History for lag construction only. Short gaps (<= history_fill_limit) may be filled;
    # long gaps remain NaN.
    history_series = ridge_history_series(
        train=train,
        material=material,
        fill_limit=history_fill_limit,
    )

    # Input history for exogenous lag features. The future/test
    # block's Masse values are intentionally never read here. Short gaps
    # may be forward-filled for feature construction only.
    input_history_series = None
    if input_lags or input_rolling:
        if "Masse" not in train.columns:
            raise ValueError(
                "Ridge Input features require 'Masse' in the prepared training data."
            )
        input_history_series = (
            pd.to_numeric(train["Masse"], errors="coerce")
            .astype(float)
            .ffill(limit=history_fill_limit)
        )

    df_model = pd.DataFrame({"y": y_raw})

    for lag in lags:
        df_model[f"lag_{lag}"] = history_series.shift(lag)

    # Rolling means are based only on values available before time t.
    # shift(1) prevents using the current target as one of its own features.
    for window in rolling_windows:
        df_model[f"rolling_mean_{window}"] = (
            history_series
            .shift(1)
            .rolling(window=window, min_periods=window)
            .mean()
        )

    # Lagged Input/Masse features. These are historical exogenous
    # observations and are safe for training. At forecast time the same
    # lag definition is reconstructed only from pre-origin Input history.
    input_feature_cols = []
    for lag in input_lags:
        col = f"input_lag_{lag}"
        df_model[col] = input_history_series.shift(lag)
        input_feature_cols.append(col)

    # Lagged rolling summaries of Input/Masse. For example, (10, 5) means
    # the mean of five historical Input values ending exactly at lag 10:
    # mean(Input[t-14], ..., Input[t-10]) when predicting target y[t].
    for lag, window in input_rolling:
        col = f"input_mean_{window}_lag_{lag}"
        df_model[col] = (
            input_history_series
            .shift(lag)
            .rolling(window=window, min_periods=window)
            .mean()
        )
        input_feature_cols.append(col)

    # Calendar effect: encode Tuesday-Friday by one-hot encoding as dummy variables.
    # Monday (day_of_week=0) is the reference category. Use the prepared
    # calendar column from the dataset rather than recomputing it here.
    dow_feature_cols = []
    if use_day_of_week:
        if "day_of_week" not in train.columns:
            raise ValueError(
                "Ridge day-of-week feature requires 'day_of_week' in the prepared training data."
            )

        train_day_of_week = pd.to_numeric(
            train["day_of_week"],
            errors="coerce",
        )

        for dow in range(1, 5):
            col = f"dow_{dow}"
            df_model[col] = (train_day_of_week == dow).astype(int).to_numpy()
            dow_feature_cols.append(col)

    holiday_feature_cols = []
    if use_holiday:
        if "is_holiday" not in train.columns:
            raise ValueError(
                "Ridge holiday feature requires 'is_holiday' in the prepared training data."
            )

        df_model["is_holiday"] = (
            pd.to_numeric(train["is_holiday"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        holiday_feature_cols.append("is_holiday")

    school_holiday_feature_cols = []
    if use_school_holiday:
        if "is_school_holiday" not in train.columns:
            raise ValueError(
                "Ridge school-holiday feature requires 'is_school_holiday' in the prepared training data."
            )

        df_model["is_school_holiday"] = (
            pd.to_numeric(train["is_school_holiday"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        school_holiday_feature_cols.append("is_school_holiday")

    feature_cols = (
        [f"lag_{lag}" for lag in lags]
        + [f"rolling_mean_{window}" for window in rolling_windows]
        + input_feature_cols
        + dow_feature_cols
        + holiday_feature_cols
        + school_holiday_feature_cols
    )

    # This removes:
    # - rows with an originally missing target,
    # - initial rows where a lag cannot yet be constructed, and
    # - rows affected by gaps that remain missing after limited ffill.
    df_model = df_model.dropna(subset=["y", *feature_cols])

    if df_model.empty:
        return np.full(horizon, np.nan)

    X_train = df_model[feature_cols]
    y_train = df_model["y"]

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Use the same history at forecast origin. Do not bfill missing (data leakage)
    # values from later dates.
    history = history_series.tolist()
    input_history = (
        input_history_series.tolist()
        if input_history_series is not None
        else None
    )
    predictions = []

    # Future-known calendar features come directly from the prepared future
    # block (the current CV test block or forecasting block). Their values are
    # known at forecast time and do not use future target observations.
    if use_day_of_week or use_holiday or use_school_holiday:
        if future is None:
            raise ValueError(
                "Ridge calendar features require the prepared future/test block."
            )
        if len(future) != horizon:
            raise ValueError(
                f"Ridge future block length ({len(future)}) must equal horizon ({horizon})."
            )

    if use_day_of_week and "day_of_week" not in future.columns:
        raise ValueError(
            "Ridge day-of-week feature requires 'day_of_week' in the prepared future data."
        )

    if use_holiday and "is_holiday" not in future.columns:
        raise ValueError(
            "Ridge holiday feature requires 'is_holiday' in the prepared future data."
        )

    if use_school_holiday and "is_school_holiday" not in future.columns:
        raise ValueError(
            "Ridge school-holiday feature requires 'is_school_holiday' in the prepared future data."
        )

    for step in range(horizon):

        lag_values = [history[-lag] for lag in lags]

        rolling_values = []
        for window in rolling_windows:
            rolling_history = history[-window:]

            # Keep the same missing-value contract used for lag features:
            # if the complete required history is unavailable, the feature
            # cannot be formed without introducing an extra imputation rule.
            if (
                len(rolling_history) < window
                or not np.all(np.isfinite(rolling_history))
            ):
                return np.full(horizon, np.nan)

            rolling_values.append(float(np.mean(rolling_history)))

        input_values = []
        for lag in input_lags:
            # For step=0 use Input at origin-lag; for later steps move
            # forward through the already observed pre-origin Input history.
            # Because lag >= horizon, this index never reaches future Input.
            input_index = len(input_history) - lag + step
            if input_index < 0 or input_index >= len(input_history):
                return np.full(horizon, np.nan)

            input_value = input_history[input_index]
            if not np.isfinite(input_value):
                return np.full(horizon, np.nan)

            input_values.append(float(input_value))

        for lag, window in input_rolling:
            # Forecast row n+step uses a historical Input window ending at
            # n+step-lag. Because lag >= horizon, even the last forecast step
            # ends no later than the forecast origin and remains leakage-free.
            input_end = len(input_history) - lag + step
            input_start = input_end - window + 1

            if input_start < 0 or input_end >= len(input_history):
                return np.full(horizon, np.nan)

            input_window = input_history[input_start:input_end + 1]
            if (
                len(input_window) < window
                or not np.all(np.isfinite(input_window))
            ):
                return np.full(horizon, np.nan)

            input_values.append(float(np.mean(input_window)))

        calendar_values = []
        if use_day_of_week:
            forecast_dow = pd.to_numeric(
                pd.Series([future["day_of_week"].iloc[step]]),
                errors="coerce",
            ).iloc[0]
            calendar_values = [
                int(forecast_dow == dow)
                for dow in range(1, 5)
            ]

        holiday_values = []
        if use_holiday:
            forecast_holiday = pd.to_numeric(
                pd.Series([future["is_holiday"].iloc[step]]),
                errors="coerce",
            ).iloc[0]
            holiday_values = [float(forecast_holiday)]

        school_holiday_values = []
        if use_school_holiday:
            forecast_school_holiday = pd.to_numeric(
                pd.Series([future["is_school_holiday"].iloc[step]]),
                errors="coerce",
            ).iloc[0]
            school_holiday_values = [float(forecast_school_holiday)]

        feature_values = (
            lag_values
            + rolling_values
            + input_values
            + calendar_values
            + holiday_values
            + school_holiday_values
        )

        # If a required lag is still unavailable because of a long gap,
        # a valid Ridge prediction cannot be formed for this block.
        if not np.all(np.isfinite(feature_values)):
            return np.full(horizon, np.nan)

        features = pd.DataFrame(
            [feature_values],
            columns=feature_cols,
        )

        y_pred = float(model.predict(features)[0])

        predictions.append(y_pred)
        history.append(y_pred)

    return np.asarray(predictions, dtype=float)

# Get model parameters from training/ forecasting config
def get_model_params(
    config: dict,
    model_name: str,
    horizon: int,
    material: str,
) -> dict:

    model_config = config["models"][model_name]

    if not model_config:
        return {}

    return model_config[horizon][material]

MODELS = {
    "naive": forecast_naive,
    "moving_average": forecast_moving_average,
    "arima/sarima": forecast_sarimax,
    "ridge": forecast_ridge,
}