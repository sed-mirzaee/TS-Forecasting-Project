from pathlib import Path

import pandas as pd
from pmdarima import auto_arima


# ============================================================
# Missing-value handling
# ============================================================

"""
Impute missing values without removing time points.

The interpolation is performed on Business-Day positions,
not calendar-day distances. Therefore, lag 5 continues to
represent five Business Days.

Parameters
----------
method:
    One of:
    - "linear"
    - "pchip"
    - "spline"

Returns
-------
filled_series:
    Complete time series with the original index.

diagnostics:
    Information about the imputation.
"""

def impute_for_autoarima(
    series: pd.Series,
    method: str = "linear",
    spline_order: int = 3,
) -> tuple[pd.Series, dict]:

    valid_methods = {
        "linear",
        "pchip",
        "akima",
        "spline",
        "polynomial",
    }

    if method not in valid_methods:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            f"Choose one of {sorted(valid_methods)}."
        )

    original_index = series.index
    missing_before = int(series.isna().sum())

    # Use row positions so every step represents one Business Day.
    positional_series = pd.Series(
        data=series.to_numpy(dtype=float),
        index=pd.RangeIndex(len(series)),
        name=series.name,
    )

    if method in {"spline", "polynomial"}:
        filled = positional_series.interpolate(
            method=method,
            order=spline_order,
            limit_direction="both",
        )
    else:
        filled = positional_series.interpolate(
            method=method,
            limit_direction="both",
        )

    # Some nonlinear methods may leave boundary values missing.
    # These fallbacks preserve the number of rows.
    filled = filled.ffill().bfill()

    missing_after = int(filled.isna().sum())

    if missing_after > 0:
        raise ValueError(
            f"Material {series.name} still contains "
            f"{missing_after} missing values after imputation."
        )

    # Check whether nonlinear interpolation produced invalid fractions.
    outside_domain = int(
        ((filled < 0.0) | (filled > 1.0)).sum()
    )

    # Material fractions must remain inside [0, 1].
    filled = filled.clip(lower=0.0, upper=1.0)

    # Restore the original Date index.
    filled.index = original_index
    filled.name = series.name

    if len(filled) != len(series):
        raise ValueError(
            f"Series length changed for Material {series.name}."
        )

    diagnostics = {
        "imputation_method": method,
        "spline_order": (
            spline_order
            if method in {"spline", "polynomial"}
            else None
        ),
        "n_missing_before": missing_before,
        "n_missing_after": missing_after,
        "n_outside_domain_before_clip": outside_domain,
        "n_observations": len(filled),
    }

    return filled, diagnostics


# ============================================================
# AutoARIMA candidate selection
# ============================================================

"""
Run AutoARIMA twice for every material:

1. Non-seasonal ARIMA
2. Seasonal SARIMA with m=5

Only the initial training period is used. Missing values
are imputed without deleting Business-Day rows.
"""
def select_arima_candidates(
    df_working: pd.DataFrame,
    materials: list[str],
    initial_ratio: float,
    output_dir: Path,
    imputation_method: str = "linear",
    spline_order: int = 3,
    seasonal_period: int = 5,
    trace: bool = False,
) -> tuple[dict, pd.DataFrame]:


    if not 0 < initial_ratio < 1:
        raise ValueError(
            "initial_ratio must be between 0 and 1."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure chronological order.
    if "Date" in df_working.columns:
        data = (
            df_working
            .sort_values("Date")
            .set_index("Date")
            .copy()
        )
    else:
        data = df_working.sort_index().copy()

    if data.index.has_duplicates:
        raise ValueError(
            "The time-series index contains duplicate dates."
        )

    initial_train_size = int(
        len(data) * initial_ratio
    )

    if initial_train_size < 20:
        raise ValueError(
            "The initial training period is too short."
        )

    candidates = {}
    results = []

    for material in materials:

        print()
        print("=" * 70)
        print(f"AutoARIMA selection: Material {material}")
        print("=" * 70)

        # Split before imputation to prevent leakage.
        initial_series = (
            data[material]
            .astype(float)
            .iloc[:initial_train_size]
            .copy()
        )

        selection_series, imputation_info = (
            impute_for_autoarima(
                series=initial_series,
                method=imputation_method,
                spline_order=spline_order,
            )
        )

        print(
            f"Imputation method: "
            f"{imputation_info['imputation_method']}"
        )
        print(
            f"Missing values imputed: "
            f"{imputation_info['n_missing_before']}"
        )
        print(
            f"Values outside [0, 1] before clipping: "
            f"{imputation_info['n_outside_domain_before_clip']}"
        )
        print(
            f"Observations retained: "
            f"{imputation_info['n_observations']}"
        )

        candidates[material] = {}

        model_settings = {
            "arima": {
                "seasonal": False,
                "m": 1,
            },
            "sarima": {
                "seasonal": True,
                "m": seasonal_period,
            },
        }

        for candidate_name, settings in model_settings.items():

            print()
            print(
                f"Material {material} | "
                f"{candidate_name.upper()}"
            )

            try:
                model_arguments = {
                    "y": selection_series,

                    # Non-seasonal orders
                    "start_p": 0,
                    "start_q": 0,
                    "max_p": 5,
                    "max_q": 5,
                    "d": None,
                    "max_d": 2,

                    # Seasonal settings
                    "seasonal": settings["seasonal"],
                    "m": settings["m"],

                    # Selection settings
                    "information_criterion": "aicc",
                    "stepwise": True,

                    # Execution
                    "suppress_warnings": True,
                    "error_action": "ignore",
                    "trace": trace,
                }

                if settings["seasonal"]:
                    model_arguments.update({
                        "start_P": 0,
                        "start_Q": 0,
                        "max_P": 2,
                        "max_Q": 2,
                        "D": None,
                        "max_D": 1,
                    })

                auto_model = auto_arima(
                    **model_arguments
                )

                order = tuple(auto_model.order)
                seasonal_order = tuple(
                    auto_model.seasonal_order
                )

                aic = float(auto_model.aic())
                aicc = float(auto_model.aicc())
                bic = float(auto_model.bic())

                candidates[material][candidate_name] = {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "aic": aic,
                    "aicc": aicc,
                    "bic": bic,
                }

                results.append({
                    "material": material,
                    "candidate": candidate_name,
                    "status": "success",
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "seasonal_period": settings["m"],
                    "aic": aic,
                    "aicc": aicc,
                    "bic": bic,
                    "imputation_method": (
                        imputation_info["imputation_method"]
                    ),
                    "spline_order": (
                        imputation_info["spline_order"]
                    ),
                    "n_observations": (
                        imputation_info["n_observations"]
                    ),
                    "n_missing_imputed": (
                        imputation_info["n_missing_before"]
                    ),
                    "n_outside_domain_before_clip": (
                        imputation_info[
                            "n_outside_domain_before_clip"
                        ]
                    ),
                    "error": None,
                })

                print(f"order          = {order}")
                print(
                    f"seasonal_order = {seasonal_order}"
                )
                print(f"AICc           = {aicc:.6f}")

            except Exception as error:

                candidates[material][candidate_name] = None

                results.append({
                    "material": material,
                    "candidate": candidate_name,
                    "status": "failed",
                    "order": None,
                    "seasonal_order": None,
                    "seasonal_period": settings["m"],
                    "aic": None,
                    "aicc": None,
                    "bic": None,
                    "imputation_method": (
                        imputation_info["imputation_method"]
                    ),
                    "spline_order": (
                        imputation_info["spline_order"]
                    ),
                    "n_observations": (
                        imputation_info["n_observations"]
                    ),
                    "n_missing_imputed": (
                        imputation_info["n_missing_before"]
                    ),
                    "n_outside_domain_before_clip": (
                        imputation_info[
                            "n_outside_domain_before_clip"
                        ]
                    ),
                    "error": str(error),
                })

                print(
                    f"{candidate_name.upper()} failed: "
                    f"{error}"
                )

    results_df = pd.DataFrame(results)

    # Mark the lower-AICc candidate for each material.
    results_df["best_by_aicc"] = False

    successful_results = results_df[
        (results_df["status"] == "success")
        & results_df["aicc"].notna()
    ]

    if not successful_results.empty:
        best_indices = (
            successful_results
            .groupby("material")["aicc"]
            .idxmin()
        )

        results_df.loc[
            best_indices,
            "best_by_aicc",
        ] = True

    # Numeric sorting for material names "1", ..., "13".
    results_df["_material_number"] = (
        results_df["material"].astype(int)
    )

    results_df = (
        results_df
        .sort_values(
            ["_material_number", "candidate"]
        )
        .drop(columns="_material_number")
        .reset_index(drop=True)
    )

    output_file = (
        output_dir
        / f"autoarima_candidates_{imputation_method}.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("AutoARIMA candidate selection completed.")
    print(f"Results saved to: {output_file}")
    print("=" * 70)

    return candidates, results_df