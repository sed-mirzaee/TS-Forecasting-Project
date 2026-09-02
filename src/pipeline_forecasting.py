import json
import time

import numpy as np
import pandas as pd

from src.config import FORECASTING_DATA_FILE, FORECASTING_OUTPUT_DIR, MATERIALS
from src.data_loader import load_forecasting_data
from src.forecasting import MODELS
from src.evaluation import smape

def run_forecasting(config: dict) -> None:

    # Load prepared forecasting data
    df = load_forecasting_data(FORECASTING_DATA_FILE)
    df = df.sort_values("Date").reset_index(drop=True)

    forecast_start = pd.Timestamp(config["forecast_start"])
    forecast_end = pd.Timestamp(config["forecast_end"])

    # Prepare train and test parts of data
    history_base = df.loc[df["Date"] < forecast_start].copy()
    test_base = df.loc[(df["Date"] >= forecast_start) & (df["Date"] <= forecast_end)].copy()

    forecast_results = []
    fold_results = []
    fold_summary = []
    summary = []

    start_time = time.perf_counter()

    for horizon in config["horizons"]:

        P_ks = []
        history = history_base.copy()
        P_ks = []
        start = 0
        fold = 1

        # This loop change to use forecast start
        while start < len(test_base):
            end = min(start + horizon, len(test_base))
            test_block = test_base.iloc[start:end].copy()
            material_scores = []

            for material in config["materials"]:

                # Freezed Parameters
                model_info = config["selected_models"][horizon][material]
                model_name = model_info["model"]
                model_params = model_info.get("params", {})
                forecast_function = MODELS[model_name]

                observed_mask = test_block[material].notna()

                forecast_kwargs = {
                    "train": history,
                    "material": material,
                    "horizon": len(test_block),
                    **model_params,
                }

                if model_name == "ridge":
                    forecast_kwargs["future"] = test_block

                prediction = forecast_function(**forecast_kwargs)

                # Keep forecasted values
                for i in range(len(test_block)):
                    forecast_results.append({
                        "model": model_name,
                        "horizon": horizon,
                        "fold": fold,
                        "material": material,
                        "Date": test_block["Date"].iloc[i],
                        "actual": test_block[material].iloc[i],
                        "forecast": prediction[i],
                        "is_observed": bool(observed_mask.iloc[i]),
                    })

                # Removing missed values from error computing
                if observed_mask.any():
                    y_true = test_block.loc[observed_mask, material].to_numpy()
                    y_pred = prediction[observed_mask.to_numpy()]
                    score = smape(y_true, y_pred)
                else:
                    score = np.nan

                material_scores.append(score)

                fold_results.append({
                    "model": model_name,
                    "horizon": horizon,
                    "fold": fold,
                    "material": material,
                    "train_end": history["Date"].max(),
                    "test_start": test_block["Date"].min(),
                    "test_end": test_block["Date"].max(),
                    "n_test": len(test_block),
                    "n_observed": int(observed_mask.sum()),
                    "smape": score,
                })

            # When we missed all values in forecast horizon, none of materials has score!
            if np.isnan(material_scores).all():
                P_k = np.nan
            else:
                P_k = np.nanmean(material_scores)

            P_ks.append(P_k)

            fold_summary.append({
                "horizon": horizon,
                "fold": fold,
                "test_start": test_block["Date"].min(),
                "test_end": test_block["Date"].max(),
                "n_test": len(test_block),
                "P_k": P_k,
            })

            history = pd.concat([history, test_block], ignore_index=True)

            start = end
            fold += 1

        Score = np.nansum(P_ks)

        summary.append({
            "Horizon": horizon,
            "n_folds": len(P_ks),
            "n_evaluated_folds": np.sum(~np.isnan(P_ks)),  # Removes folds that are completely missing value
            "Score": Score,
            "mean_P_k": np.nanmean(P_ks),
        })

        # Save prediction for each horizon as project definition
        save_horizon_forecast(forecast_results, horizon)

    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} seconds")

    # Save global sMAPE (Score) for each horizon as project definition
    save_global_smape(summary)

    # Save summaries and details results for figures and report
    save_forecasting_results(
        config,
        forecast_results,
        fold_results,
        fold_summary,
        summary
    )

def save_horizon_forecast(forecast_results, horizon):
    forecast_dir = FORECASTING_OUTPUT_DIR / "submission"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    df_forecasts = pd.DataFrame(forecast_results)

    df_horizon = df_forecasts[df_forecasts["horizon"] == horizon].copy()

    df_csv = df_horizon.pivot(
        index="Date",
        columns="material",
        values="forecast"
    ).reset_index()

    # rename & sort columns
    df_csv = df_csv.rename(columns={material: f"Forecast_Fraction_{material}" for material in MATERIALS})
    columns = ["Date", *[f"Forecast_Fraction_{material}" for material in MATERIALS]]
    df_csv = df_csv[columns]

    output_file = forecast_dir / f"forecast_horizon_{horizon}.csv"
    df_csv.to_csv(output_file, index=False)

def save_global_smape(summary):
    forecast_dir = FORECASTING_OUTPUT_DIR / "submission"
    forecast_dir.mkdir(parents=True, exist_ok=True)


    df_smape = pd.DataFrame(summary)
    df_smape = df_smape[["Horizon", "Score"]]

    output_file = forecast_dir / f"forecast_score.csv"
    df_smape.to_csv(output_file, index=False)

# The total Evaluation is performed by these formulas, so in this function we save detailed results and summary in each step.
# Score = Σₖ Pₖ
# k: rolling (horizon-wise) step

# Pₖ = 1/13 Σⱼ SMAPEⱼₖ
# j: material index

# SMAPEⱼₖ = 2/T Σᵢ |yᵢⱼ - ŷᵢⱼ|/max(ε, |yᵢⱼ| + |ŷᵢⱼ|)  , ε = 1/1000
# T: forecast-horizon length
#    Missing actual observations are excluded from validation scoring.
def save_forecasting_results(config, forecast_results, fold_results, fold_summary, summary):
    forecast_dir = FORECASTING_OUTPUT_DIR
    forecast_dir.mkdir(parents=True, exist_ok=True)

    # Save forecasted values
    pd.DataFrame(forecast_results).to_csv(
        forecast_dir / "forecasts.csv",
        index=False
    )

    # Save results for each material j, in specified fold k and SMAPEⱼₖ
    pd.DataFrame(fold_results).to_csv(
        forecast_dir / "fold_results.csv",
        index=False
    )

    # Save summary for specified fold k and Pₖ = 1/13 Σⱼ SMAPEⱼₖ
    pd.DataFrame(fold_summary).to_csv(
        forecast_dir / "fold_summary.csv",
        index=False
    )

    # Save summary for all horizons with Scores
    pd.DataFrame(summary).to_csv(
        forecast_dir / "summary.csv",
        index=False
    )

    with open(forecast_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Forecasting results saved to: {forecast_dir}")