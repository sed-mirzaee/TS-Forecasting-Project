# Time-Series Forecasting Project

Forecasting pipeline developed for the TU Dortmund Data Mining Cup case study with REMONDIS.

The project forecasts daily fractions for 13 materials over four business-day horizons:

- 10 business days
- 22 business days
- 66 business days
- 132 business days

The complete workflow covers data preparation, rolling time-series validation, model comparison, frozen model selection, and rolling forecasting on the 2025 test period.

## Project design

The forecasting problem is treated as 52 separate forecasting tasks:

`4 horizons × 13 materials`

A different model and configuration may therefore be selected for each material and horizon. Final model choices and hyperparameters are stored in `src/config_forecasting.py` and must remain frozen during the final test evaluation.

The considered model families are:

- Naive forecast
- Moving-average baseline
- ARIMA / SARIMA
- Ridge regression with time-series, calendar, and Input-derived features

## Data preparation

The preprocessing pipeline follows these rules:

- Standardize and validate dates and required columns.
- Replace invalid negative Input values with missing values.
- Treat long zero blocks in Input according to the configured cleaning threshold.
- Preserve valid zero-valued material fractions.
- Validate that observed material fractions lie in `[0, 1]` and sum approximately to 1.
- Align data to the complete calendar before restricting the modeling data to Monday-Friday business days.
- Keep weekday public holidays in the business-day timeline.
- Preserve missing observations instead of silently normalizing or globally imputing them.
- Merge calendar/external features only after the base data have been cleaned and aligned.
- Apply the project start date `2020-03-23` after cleaning/alignment/merge and before the prepared training data are saved.

Training data end on `2024-12-31`. The 2025 data are used only by the rolling forecasting stage.

## Missing-value contract

Missing values are handled inside the model workflow rather than by globally filling the prepared dataset.

For evaluation:

- The original target values are retained.
- Scores are calculated only for originally observed target values.
- A completely missing test block receives no material score, but the rolling time index still advances.

For Ridge:

- The train/test split is created before model-specific missing-value handling.
- The observed target remains the real target; imputed values are not used as artificial training targets.
- A separate history series may be prepared inside the training fold for constructing lag and rolling features.
- Missing values created naturally by lag/rolling feature engineering are dropped rather than imputed.
- No future test observations are used to construct training features.

## Rolling validation and forecasting

All horizons use expanding/rolling time-series evaluation.

For a horizon `h`:

1. Fit the selected model using only the history available at the start of the block.
2. Forecast the next `h` business days.
3. Evaluate only observations that are actually available.
4. After all forecasts for the current block have been generated, append the newly observed block to the history.
5. Refit the same frozen model configuration for the next block.

Model families, feature sets, orders, regularization strengths, rolling windows, and all other hyperparameters must not be changed during the final 2025 evaluation.

## Repository structure

The core project is organized around the following modules/files:

```text
.
├── src/
│   ├── config.py
│   ├── config_forecasting.py
│   ├── data_loader.py
│   ├── preprocessing_common.py
│   ├── preprocessing_model_based.py
│   ├── evaluation.py
│   ├── forecasting.py
│   ├── pipeline_data.py
│   ├── pipeline_training.py
│   └── pipeline_forecasting.py
├── data/
├── outputs/
├── requirements.txt
└── README.md
```

Additional analysis notebooks/scripts may be present for EDA and model-selection experiments. They are not part of the frozen final forecasting configuration unless referenced by the production pipeline.

## Environment setup

Python 3.10+ is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Recommended execution order

Run the project in 4 logical stages:

1. **Data preparation**  
   - Set STAGE = Stage.DATA_PREPARATION, DATA_MODE = Data_mode.TRAINING
   - Build the prepared training dataset and external calendar features.

2. **Training / validation** 
   - Set STAGE = Stage.TRAINING
   - Use the rolling cross-validation pipeline only for analysis and model comparison. Do not use this stage to modify the already frozen final configuration during final evaluation.

3. **Data preparation for forecasting** 
   - Set STAGE = Stage.DATA_PREPARATION, DATA_MODE = Data_mode.FORECASTING
   - Build the prepared forecasting dataset and external calendar features.

4. **Forecasting** 
   - Set STAGE = Stage.FORECASTING
   - Run the 2025 rolling forecasting pipeline using only the selections stored in `src/config_forecasting.py`.

## Forecast output

The competition submission requires one wide CSV per horizon with the following structure:

```text
Date,Forecast_Fraction_1,Forecast_Fraction_2,...,Forecast_Fraction_13
2025-01-01,...,...,...,...
...
```

The production forecasting pipeline may additionally save internal audit files such as fold-level scores, detailed forecasts, summaries, and a configuration snapshot. These files are for reproducibility and are not substitutes for the required wide submission files.

## Frozen configuration

The final configuration is defined per `(horizon, material)` in:

```text
src/config_forecasting.py
```

- 4 horizons
- 13 materials per horizon
- 52 complete model selections
- all parameters required by the selected forecasting function

## Data confidentiality

Raw project data are confidential.
