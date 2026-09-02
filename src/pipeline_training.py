import json
import time

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import TRAINING_DATA_FILE, MATERIALS, TRAINING_OUTPUT_DIR, INITIAL_RATIO, OUTPUT_DIR, HORIZONS
from src.data_loader import load_training_data
from src.cross_validation import expanding_cv
from src.model_selection import select_arima_candidates
from src.forecasting import MODELS, get_model_params, forecast_sarimax
from src.evaluation import smape

def run_training(config: dict) -> None:

    print(f"Starting training run: {config['run_name']}")

    # Load prepared training data
    df = load_training_data(TRAINING_DATA_FILE)

    forecast_results = []
    fold_results = []
    model_summary = []

    start_time = time.perf_counter()

    for horizon in config["horizons"]:

        for model_name, candidates in config["models"].items():

            forecast_function = MODELS[model_name]

            for candidate in candidates:

                candidate_name = candidate["name"]
                model_params = dict(candidate.get("params", {}))

                candidate_materials = candidate.get("materials", "all")
                if candidate_materials == "all":
                    candidate_materials = MATERIALS

                for material in candidate_materials:

                    model_fold_scores = []
                    n_expected_folds = 0

                    for fold, (train, test) in enumerate(
                            expanding_cv(
                                df,
                                horizon=horizon,
                                initial_ratio=config["initial_ratio"]
                            ),
                            start=1
                    ):

                        observed_mask = test[material].notna()

                        forecast_kwargs = {
                            "train": train,
                            "material": material,
                            "horizon": len(test),
                            **model_params,
                        }

                        if model_name == "ridge":
                            forecast_kwargs["future"] = test # just used holiday and doy of week, ...

                        prediction = forecast_function(**forecast_kwargs)

                        # Removing missed values from error computing
                        if observed_mask.any():
                            n_expected_folds += 1
                            y_true = test.loc[observed_mask, material].to_numpy()
                            y_pred = prediction[observed_mask.to_numpy()]
                            score = smape(y_true, y_pred)
                        else:
                            score = np.nan

                        model_fold_scores.append(score)

                        fold_results.append({
                            "horizon": horizon,
                            "material": material,
                            "model": candidate_name,
                            "fold": fold,
                            "train_end": train["Date"].max(),
                            "test_start": test["Date"].min(),
                            "test_end": test["Date"].max(),
                            "n_test": len(test),
                            "n_observed": int(observed_mask.sum()),
                            "smape": score,
                        })

                        # Keep forecasted values
                        for i in range(len(test)):
                            forecast_results.append({
                                "horizon": horizon,
                                "material": material,
                                "model": candidate_name,
                                "fold": fold,
                                "Date": test["Date"].iloc[i],
                                "actual": test[material].iloc[i],
                                "forecast": prediction[i],
                                "is_observed": bool(observed_mask.iloc[i]),
                            })

                    finite_scores = np.asarray(
                        model_fold_scores,
                        dtype=float,
                    )

                    model_summary.append({
                        "horizon": horizon,
                        "material": material,
                        "model": candidate_name,
                        "mean_smape": np.nanmean(finite_scores),
                        "std_smape": np.nanstd(finite_scores),
                        "sum_smape": np.nansum(finite_scores),
                        "n_expected_folds": n_expected_folds,
                        "n_evaluated_folds": int(np.isfinite(finite_scores).sum()),
                    })

    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} seconds")

    save_training_results(
        config,
        forecast_results,
        fold_results,
        model_summary
    )

# The total Evaluation is performed by these formulas, but in training phase for each material and candidate, one SMAPE is calculated.

# Score = Σₖ Pₖ
# k: rolling (horizon-wise) step

# Pₖ = 1/13 Σⱼ SMAPEⱼₖ
# j: material index

# SMAPEⱼₖ = 2/T Σᵢ |yᵢⱼ - ŷᵢⱼ|/max(ε, |yᵢⱼ| + |ŷᵢⱼ|)  , ε = 1/1000
# T: forecast-horizon length
#    Missing actual observations are excluded from validation scoring.

def save_training_results(config, forecast_results, fold_results, model_summary, ):
    run_dir = TRAINING_OUTPUT_DIR / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save forecasted values
    pd.DataFrame(forecast_results).to_csv(
        run_dir / "forecasts.csv",
        index=False
    )

    # Save results for each material j, in specified fold k and SMAPEⱼₖ
    pd.DataFrame(fold_results).to_csv(
        run_dir / "fold_results.csv",
        index=False
    )

    # Save summary for specified model, one SMAPE for each material and candidate model
    pd.DataFrame(model_summary).to_csv(
        run_dir / "model_summary.csv",
        index=False
    )

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Training results saved to: {run_dir}")

def save_arima_training_results(run_name, forecast_results, fold_results, candidate_summary):
    run_dir = TRAINING_OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save forecasted values
    pd.DataFrame(forecast_results).to_csv(
        run_dir / "forecasts.csv",
        index=False
    )

    # Save results for each material j, in specified fold k and SMAPEⱼₖ
    pd.DataFrame(fold_results).to_csv(
        run_dir / "fold_results.csv",
        index=False
    )

    # Save summary for specified candidate and mean(SMAPEⱼₖ) and sum(SMAPEⱼₖ)
    pd.DataFrame(candidate_summary).to_csv(
        run_dir / "candidate_summary.csv",
        index=False
    )

    print(f"Training results for all candidates saved to: {run_dir}")

def autoarima_selection_pipeline() -> None:

    data = load_training_data(TRAINING_DATA_FILE)

    df_working = data.copy()

    df_working = (
        df_working
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print("Running AutoARIMA candidate selection...")

    linear_candidates, linear_results = (
        select_arima_candidates(
            df_working=df_working,
            materials=MATERIALS,
            initial_ratio=INITIAL_RATIO,
            output_dir=Path(OUTPUT_DIR) / "autoarima",
            imputation_method="linear",
            seasonal_period=5,
            trace=True,
        )
    )

    pchip_candidates, pchip_results = (
        select_arima_candidates(
            df_working=df_working,
            materials=MATERIALS,
            initial_ratio=INITIAL_RATIO,
            output_dir=Path(OUTPUT_DIR) / "autoarima",
            imputation_method="pchip",
            seasonal_period=5,
            trace=True,
        )
    )

    spline_candidates, spline_results = (
        select_arima_candidates(
            df_working=df_working,
            materials=MATERIALS,
            initial_ratio=INITIAL_RATIO,
            output_dir=Path(OUTPUT_DIR) / "autoarima",
            imputation_method="spline",
            spline_order=3,
            seasonal_period=5,
            trace=True,
        )
    )

def run_arima_training(config: dict) -> None:

    print(f"Starting training arima/sarima to find best order:")

    # Load prepared training data
    df = load_training_data(TRAINING_DATA_FILE)

    forecast_results = []
    fold_results = []
    candidate_results = []

    start_time = time.perf_counter()

    for horizon in HORIZONS:

        for material in MATERIALS:

            material_candidates = (config[material])

            for candidate in material_candidates:

                candidate_name = candidate["name"]
                order = candidate["order"]
                seasonal_order = (candidate["seasonal_order"])

                candidate_fold_scores = []

                for fold, (train, test) in enumerate(
                        expanding_cv(
                            df,
                            horizon=horizon,
                            initial_ratio=INITIAL_RATIO
                        ),
                        start=1
                ):

                    # Missing values in test must not be evaluated.
                    observed_mask = test[material].notna()

                    prediction = forecast_sarimax(
                        train=train,
                        material=material,
                        horizon=len(test),
                        order=order,
                        seasonal_order=(
                            seasonal_order
                        ),
                    )

                    if observed_mask.any():
                        y_true = test.loc[observed_mask, material].to_numpy()
                        y_pred = prediction[observed_mask.to_numpy()]
                        score = smape(y_true, y_pred)
                    else:
                        score = np.nan

                    candidate_fold_scores.append(score)

                    fold_results.append({
                        "material": material,
                        "horizon": horizon,
                        "candidate": candidate_name,
                        "fold": fold,
                        "order": str(order),
                        "seasonal_order": str(
                            seasonal_order
                        ),
                        "sMAPE": score,
                    })

                    for i in range(len(test)):
                        forecast_results.append({
                            "material": material,
                            "horizon": horizon,
                            "candidate": candidate_name,
                            "fold": fold,
                            "Date": test["Date"].iloc[i],
                            "actual": test[material].iloc[i],
                            "forecast": prediction[i],
                            "is_observed": bool(
                                observed_mask.iloc[i]
                            ),
                        })

                # This candidate has now completed all folds.
                if np.isnan(candidate_fold_scores).all():
                    mean_smape = np.nan
                    sum_smape = np.nan
                else:
                    mean_smape = np.nanmean(candidate_fold_scores)
                    sum_smape = np.nansum(candidate_fold_scores) # as defined in problem definition

                candidate_results.append({
                    "material": material,
                    "horizon": horizon,
                    "candidate": candidate_name,
                    "order": str(order),
                    "seasonal_order": str(
                        seasonal_order
                    ),
                    "mean_smape": mean_smape,
                    "sum_smape": sum_smape,
                    "number_of_folds": int(
                        np.isfinite(
                            candidate_fold_scores
                        ).sum()
                    ),
                })

    candidate_results_df = pd.DataFrame(candidate_results)
    valid_results = candidate_results_df[candidate_results_df["mean_smape"].notna()].copy()

    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} seconds")

    best_candidates = (
        valid_results.loc[
            valid_results
            .groupby(
                ["material", "horizon"]
            )["mean_smape"]
            .idxmin()
        ]
            .sort_values(
            ["horizon", "material"]
        )
            .reset_index(drop=True)
    )

    print(best_candidates)

    save_arima_training_results(
        "arima_sarima_v02",
        forecast_results,
        fold_results,
        candidate_results,
    )
