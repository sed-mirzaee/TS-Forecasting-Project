# ============================================================
# Data Loading methods
# ============================================================

from pathlib import Path
import pandas as pd
import requests

"""
Load the raw Input and Material Fraction datasets.

Parameters
----------
input_path : Path
    Path to the raw Input dataset.

material_path : Path
    Path to the raw Material Fraction dataset.
"""
def load_raw_data(input_path: Path, material_path: Path, ) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_input = pd.read_csv(input_path)
    df_material = pd.read_csv(material_path)

    return df_input, df_material

def load_forecast_raw_data( mode: str, input_path: Path, material_path: Path, input_forecast: Path, material_forecast: Path,) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_input_new = pd.read_csv(input_forecast)
    df_material_new = pd.read_csv(material_forecast)

    if mode == "FULL":
        return df_input_new, df_material_new

    if mode == "APPEND":
        df_input_old, df_material_old = load_raw_data(input_path, material_path)

        df_input = pd.concat([df_input_old, df_input_new], ignore_index=True)

        df_material = pd.concat([df_material_old, df_material_new], ignore_index=True)

        return df_input, df_material

    raise ValueError(
        f"Unknown forecast data mode: {mode}"
    )


def load_training_data(training_data_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        training_data_path,
        parse_dates=["Date"]
    )

def load_forecasting_data(forecasting_data_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        forecasting_data_path,
        parse_dates=["Date"]
    )

"""
Get the holidays from OpenHolidays API and save in a csv file.

Parameters
----------
holidays_path : Path
    Path to save Holidays dataset.
"""
def get_public_holidays(holidays_path: Path) -> pd.DataFrame:
    all_holidays = []

    for year in range(2018, 2026):
        url = (
            f"https://openholidaysapi.org/PublicHolidays"
            f"?countryIsoCode=DE"
            f"&languageIsoCode=EN"
            f"&validFrom={year}-01-01"
            f"&validTo={year}-12-31"
            f"&subdivisionCode=DE-NW"
        )

        response = requests.get(url)
        response.raise_for_status()
        all_holidays.extend(response.json())

    df = pd.DataFrame(all_holidays)

    df["startDate"] = pd.to_datetime(df["startDate"])
    df["endDate"] = pd.to_datetime(df["endDate"])

    rows = []

    for _, row in df.iterrows():
        for date in pd.date_range(row["startDate"], row["endDate"], freq="D"):
            rows.append({
                "Date": date,
                "holiday_name": row["name"],
                "is_holiday": 1
            })

    df_holidays_aligned = pd.DataFrame(rows)

    df_holidays_aligned.to_csv(holidays_path, index=False)
    return df_holidays_aligned


def load_school_holidays(school_holiday_path: Path) -> pd.DataFrame:
    df = pd.read_csv(school_holiday_path)

    df["startDate"] = pd.to_datetime(df["startDate"], dayfirst=True)
    df["endDate"] = pd.to_datetime(df["endDate"], dayfirst=True)

    rows = []
    for _, row in df.iterrows():
        for date in pd.date_range(row["startDate"], row["endDate"], freq="D"):
            rows.append({
                "Date": date,
                "school_holiday_name": row["name"],
                "is_school_holiday": 1
            })

    return pd.DataFrame(rows)

def save_prepared_data(df: pd.DataFrame, output_file: Path) -> None:

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Prepared data saved to: {output_file}")