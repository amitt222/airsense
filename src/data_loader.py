"""
AirSense - Data Loader Module
Author: AirSense Team
Description:
    Responsible for locating, validating, loading, and filtering the historical
    air quality dataset (city_day.csv) from Kaggle's "Air Quality Data in India".
    Includes helper routines to inspect data completeness and provide clear
    download instructions if the raw file is missing.
"""

from pathlib import Path
from typing import Any, List, Optional
import pandas as pd

# Default 8 Indian major cities focused in the project
TARGET_CITIES: List[str] = [
    "Delhi",
    "Mumbai",
    "Kolkata",
    "Chennai",
    "Bengaluru",
    "Hyderabad",
    "Ahmedabad",
    "Jaipur",
]

# Standard default path for raw dataset
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "city_day.csv"


def get_dataset_download_instructions() -> str:
    """
    Returns step-by-step download instructions for the Kaggle dataset.

    Returns:
        str: Formatted instruction text.
    """
    return (
        "\n"
        "================================================================================\n"
        "                     AIRSENSE: DATASET DOWNLOAD INSTRUCTIONS\n"
        "================================================================================\n"
        "The historical dataset 'city_day.csv' was not found in 'data/raw/'.\n\n"
        "Please follow these steps to download the official dataset:\n"
        "  1. Visit the Kaggle dataset page:\n"
        "     https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india\n"
        "  2. Click on 'Download' (or download city_day.csv directly).\n"
        "  3. Extract the downloaded archive (if zipped).\n"
        "  4. Place 'city_day.csv' into the project folder:\n"
        "     data/raw/city_day.csv\n\n"
        "Once placed, re-run this loader script to verify.\n"
        "================================================================================\n"
    )


def check_raw_data_exists(file_path: Path | str = DEFAULT_DATA_PATH) -> bool:
    """
    Checks whether the raw data file exists on disk.

    Args:
        file_path (Path | str): Path to the target CSV file.

    Returns:
        bool: True if file exists and is a file, False otherwise.
    """
    path = Path(file_path)
    return path.is_file()


def load_raw_city_day(file_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads city_day.csv into a pandas DataFrame with proper datetime parsing
    and column sanitization.

    Args:
        file_path (Path | str): Path to the city_day.csv file.

    Returns:
        pd.DataFrame: Loaded and sorted historical air quality data.

    Raises:
        FileNotFoundError: If the dataset is not found at the specified path.
    """
    path = Path(file_path)
    if not path.is_file():
        instructions = get_dataset_download_instructions()
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n{instructions}"
        )

    # Read CSV with Date parsed as datetime
    df = pd.read_csv(path, parse_dates=["Date"])

    # Clean string columns: strip leading/trailing spaces
    if "City" in df.columns:
        df["City"] = df["City"].astype(str).str.strip()
        # Standardize city name variants (e.g. Bangalore -> Bengaluru)
        city_mapping = {"Bangalore": "Bengaluru"}
        df["City"] = df["City"].replace(city_mapping)

    # Sort values chronologically by city and date
    sort_cols = [col for col in ["City", "Date"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return df


def filter_target_cities(
    df: pd.DataFrame,
    cities: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filters the dataset to only include the target cities for AirSense.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'City' column.
        cities (Optional[List[str]]): List of cities to retain. Defaults to TARGET_CITIES.

    Returns:
        pd.DataFrame: Filtered DataFrame containing records only for target cities.
    """
    if "City" not in df.columns:
        raise KeyError("Column 'City' not found in DataFrame.")

    target = cities if cities is not None else TARGET_CITIES
    filtered_df = df[df["City"].isin(target)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df


def get_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generates high-level metadata summary of the dataset for validation and viva explanation.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict[str, Any]: Summary metrics (shape, date ranges, cities, missing counts).
    """
    summary: dict[str, Any] = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "cities_present": sorted(df["City"].dropna().unique().tolist()) if "City" in df.columns else [],
        "min_date": str(df["Date"].min().date()) if "Date" in df.columns and not df["Date"].empty else None,
        "max_date": str(df["Date"].max().date()) if "Date" in df.columns and not df["Date"].empty else None,
        "missing_aqi_count": int(df["AQI"].isna().sum()) if "AQI" in df.columns else 0,
        "missing_aqi_percentage": round(float((df["AQI"].isna().mean()) * 100), 2) if "AQI" in df.columns else 0.0,
    }
    return summary


if __name__ == "__main__":
    print("[AirSense] Checking dataset availability...")
    if check_raw_data_exists(DEFAULT_DATA_PATH):
        print(f"[AirSense] Found dataset at '{DEFAULT_DATA_PATH}'. Loading...")
        raw_df = load_raw_city_day(DEFAULT_DATA_PATH)
        filtered_df = filter_target_cities(raw_df)

        print("\n--- Full Dataset Summary ---")
        full_summary = get_data_summary(raw_df)
        for k, v in full_summary.items():
            print(f"  {k}: {v}")

        print("\n--- Target Cities Filtered Summary ---")
        filtered_summary = get_data_summary(filtered_df)
        for k, v in filtered_summary.items():
            print(f"  {k}: {v}")

        print("\n--- Target Cities Record Distribution ---")
        print(filtered_df["City"].value_counts())
    else:
        print(get_dataset_download_instructions())
