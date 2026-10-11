"""
AirSense - Data Preprocessing & Cleaning Module
Author: AirSense Team
Description:
    Provides robust, reproducible data cleaning routines for Indian air quality
    time-series data. Handles sensor dropouts via temporal interpolation,
    removes pre-monitoring station dead zones, and enforces physical domain
    constraints / CPCB capping to prevent outlier distortion.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

# Core pollutant columns present in city_day.csv
POLLUTANT_COLUMNS: List[str] = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene",
]

# Physical plausible upper bounds to eliminate uncalibrated sensor anomalies
PHYSICAL_UPPER_BOUNDS: dict[str, float] = {
    "PM2.5": 1000.0,    # ug/m3
    "PM10": 1200.0,     # ug/m3
    "NO": 500.0,        # ug/m3
    "NO2": 500.0,       # ug/m3
    "NOx": 600.0,       # ppb / ug/m3
    "NH3": 500.0,       # ug/m3
    "CO": 60.0,         # mg/m3
    "SO2": 400.0,       # ug/m3
    "O3": 400.0,        # ug/m3
    "Benzene": 250.0,   # ug/m3
    "Toluene": 300.0,   # ug/m3
    "Xylene": 200.0,    # ug/m3
    "AQI": 500.0,       # CPCB NAQI official maximum scale
}

PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_city_day.csv"
)


def assign_cpcb_bucket(aqi: float) -> str:
    """
    Maps a numerical AQI value to the official Central Pollution Control Board (CPCB) category.

    CPCB NAQI Standard:
      - 0 to 50: Good
      - 51 to 100: Satisfactory
      - 101 to 200: Moderate
      - 201 to 300: Poor
      - 301 to 400: Very Poor
      - 401 to 500: Severe

    Args:
        aqi (float): Computed or predicted AQI.

    Returns:
        str: CPCB category label.
    """
    if pd.isna(aqi):
        return "Unknown"
    val = float(aqi)
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Satisfactory"
    elif val <= 200:
        return "Moderate"
    elif val <= 300:
        return "Poor"
    elif val <= 400:
        return "Very Poor"
    else:
        return "Severe"


def handle_physical_outliers(
    df: pd.DataFrame,
    bounds: Optional[dict[str, float]] = None
) -> pd.DataFrame:
    """
    Handles impossible negative readings and extreme sensor malfunctions.

    Methodology:
      1. Negative values (< 0) are physical impossibilities caused by sensor calibration drift.
         These are replaced with NaN so they can be smoothly interpolated temporally.
      2. Extreme values above physical sensor bounds (or uncalibrated readings such as
         Ahmedabad AQI = 2049) are capped at regulatory/domain limits (AQI capped at 500).
         Why capping instead of dropping? Dropping creates gaps in the daily time-series,
         breaking lag features (t-1, t-2, t-7). Capping preserves sequence continuity while
         preventing gradient explosion during ML training.

    Args:
        df (pd.DataFrame): Raw DataFrame.
        bounds (Optional[dict[str, float]]): Mapping of column to max threshold.

    Returns:
        pd.DataFrame: DataFrame with sanitized values.
    """
    df_clean = df.copy()
    thresholds = bounds if bounds is not None else PHYSICAL_UPPER_BOUNDS

    for col in thresholds:
        if col in df_clean.columns:
            # 1. Negative readings -> NaN
            neg_mask = df_clean[col] < 0
            if neg_mask.any():
                df_clean.loc[neg_mask, col] = np.nan

            # 2. Upper bounds capping (Winsorization)
            upper_limit = thresholds[col]
            df_clean[col] = df_clean[col].clip(upper=upper_limit)

    return df_clean


def trim_pre_monitoring_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trims dead-zone dates prior to station commissioning for each city.

    Rationale:
      Cities like Mumbai (active May 2018) and Kolkata (active April 2018) have rows
      dating back to 2015 filled with pure NaNs before monitoring equipment was installed.
      Attempting to backfill 3 years of non-existent sensor data generates artificial flatlines.
      Trimming each city to start from its first valid AQI reading ensures data integrity.

    Args:
        df (pd.DataFrame): Filtered DataFrame with 'City', 'Date', and 'AQI'.

    Returns:
        pd.DataFrame: Trimmed DataFrame with active monitoring records.
    """
    trimmed_dfs: List[pd.DataFrame] = []

    for city, city_df in df.groupby("City", as_index=False):
        valid_indices = city_df["AQI"].dropna().index
        if not valid_indices.empty:
            first_valid_idx = valid_indices[0]
            # Keep rows from the first valid measurement onward
            trimmed = city_df.loc[first_valid_idx:].copy()
            trimmed_dfs.append(trimmed)
        else:
            # If city has no valid AQI, retain original
            trimmed_dfs.append(city_df.copy())

    result = pd.concat(trimmed_dfs, ignore_index=True)
    result.sort_values(by=["City", "Date"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing pollutant and AQI values using time-series aware techniques.

    Strategy:
      1. Complete daily calendar re-indexing: Guarantees no missing calendar dates.
      2. Time-weighted linear interpolation: Accurately bridges brief sensor dropouts
         (e.g., 1-7 days of power cuts or routine maintenance) by following the linear trajectory.
      3. Forward fill + Backward fill: Covers edge cases at boundary dates.
      4. City-specific median fallback: Fills any residual unmonitored gases (e.g. Xylene/NH3).
      5. Re-derive AQI_Bucket from cleaned AQI using official CPCB thresholds.

    Args:
        df (pd.DataFrame): DataFrame with datetime 'Date' and 'City'.

    Returns:
        pd.DataFrame: Fully imputed and clean DataFrame.
    """
    imputed_dfs: List[pd.DataFrame] = []
    numeric_cols = [c for c in df.columns if c in POLLUTANT_COLUMNS or c == "AQI"]

    for city, city_df in df.groupby("City"):
        # Re-index to ensure every calendar day is present
        city_df = city_df.sort_values("Date").drop_duplicates(subset=["Date"]).copy()
        city_df.set_index("Date", inplace=True)

        min_date = city_df.index.min()
        max_date = city_df.index.max()
        full_idx = pd.date_range(start=min_date, end=max_date, freq="D")
        city_reindexed = city_df.reindex(full_idx)
        city_reindexed["City"] = city

        # 1. Linear interpolation on time series
        city_reindexed[numeric_cols] = city_reindexed[numeric_cols].interpolate(
            method="time", limit=14
        )

        # 2. Forward fill then backward fill for edge boundaries
        city_reindexed[numeric_cols] = city_reindexed[numeric_cols].ffill().bfill()

        # 3. If an entire column remains all NaN for this city, fill with overall median
        for col in numeric_cols:
            if city_reindexed[col].isna().all():
                city_reindexed[col] = df[col].median()
            elif city_reindexed[col].isna().any():
                city_reindexed[col] = city_reindexed[col].fillna(city_reindexed[col].median())

        city_reindexed.reset_index(inplace=True)
        city_reindexed.rename(columns={"index": "Date"}, inplace=True)
        imputed_dfs.append(city_reindexed)

    result = pd.concat(imputed_dfs, ignore_index=True)

    # Recompute CPCB AQI Bucket
    result["AQI_Bucket"] = result["AQI"].apply(assign_cpcb_bucket)
    result.sort_values(by=["City", "Date"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def clean_and_preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the entire cleaning pipeline:
      Step 1: Trim pre-monitoring void dates per city.
      Step 2: Handle negative sensor values and cap extreme physical outliers.
      Step 3: Temporal interpolation and missing value imputation.
      Step 4: Re-categorize AQI buckets per CPCB NAQI standard.

    Args:
        df (pd.DataFrame): Raw loaded DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for feature engineering and modeling.
    """
    # Step 1: Trim pre-monitoring dates
    trimmed = trim_pre_monitoring_dates(df)

    # Step 2: Handle negative values and domain capping
    capped = handle_physical_outliers(trimmed)

    # Step 3: Impute missing time series
    cleaned = impute_missing_values(capped)

    return cleaned


def save_cleaned_data(
    df: pd.DataFrame,
    output_path: Path | str = PROCESSED_DATA_PATH
) -> Path:
    """
    Saves the cleaned DataFrame to CSV.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        output_path (Path | str): Destination path.

    Returns:
        Path: Path to saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    from src.data_loader import load_raw_city_day, filter_target_cities

    print("[AirSense] Starting Data Preprocessing Pipeline...")
    raw_df = load_raw_city_day()
    target_df = filter_target_cities(raw_df)

    print(f"[AirSense] Raw target records: {len(target_df)}")
    cleaned_df = clean_and_preprocess_dataset(target_df)

    print(f"[AirSense] Cleaned records: {len(cleaned_df)}")
    print(f"[AirSense] Remaining null values in cleaned data:\n{cleaned_df.isna().sum()}")

    saved_path = save_cleaned_data(cleaned_df)
    print(f"[AirSense] Cleaned dataset successfully saved to: {saved_path}")
