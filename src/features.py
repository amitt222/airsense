"""
AirSense - Feature Engineering Module
Author: AirSense Team
Description:
    Transforms cleaned air quality time-series data into feature-rich tabular representations
    for machine learning forecasting (24h, 48h, 72h).
    Engineers:
      - Multi-day lags (1, 2, 3, 7 days) without future lookahead
      - Shifted rolling statistics (means, standard deviations, min, max)
      - Calendar & cyclical seasonality features (day-of-week, month, season, sin/cos)
      - Open-Meteo historical meteorology (temperature, humidity, precipitation, wind speed)
      - Multi-horizon forecast targets (t+1, t+2, t+3 days)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import requests

# City geographic centroids for Open-Meteo weather API queries
CITY_COORDINATES: Dict[str, Dict[str, float]] = {
    "Delhi": {"latitude": 28.6139, "longitude": 77.2090},
    "Mumbai": {"latitude": 19.0760, "longitude": 72.8777},
    "Kolkata": {"latitude": 22.5726, "longitude": 88.3639},
    "Chennai": {"latitude": 13.0827, "longitude": 80.2707},
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946},
    "Hyderabad": {"latitude": 17.3850, "longitude": 78.4867},
    "Ahmedabad": {"latitude": 23.0225, "longitude": 72.5714},
    "Jaipur": {"latitude": 26.9124, "longitude": 75.7873},
}

DEFAULT_CLEANED_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_city_day.csv"
)
DEFAULT_WEATHER_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "open_meteo_weather.csv"
)
DEFAULT_FEATURES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "features_city_day.csv"
)


def fetch_open_meteo_weather_for_city(
    city: str,
    start_date: str,
    end_date: str,
    timeout: int = 30
) -> pd.DataFrame:
    """
    Fetches daily meteorological records for a single city from the free Open-Meteo Archive API.

    Args:
        city (str): City name in CITY_COORDINATES.
        start_date (str): Format 'YYYY-MM-DD'.
        end_date (str): Format 'YYYY-MM-DD'.
        timeout (int): Network timeout in seconds.

    Returns:
        pd.DataFrame: Daily weather DataFrame with Date, City, and weather columns.
    """
    if city not in CITY_COORDINATES:
        raise ValueError(f"Unknown city coordinates for '{city}'")

    coords = CITY_COORDINATES[city]
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": (
            "temperature_2m_mean,"
            "relative_humidity_2m_mean,"
            "precipitation_sum,"
            "wind_speed_10m_max,"
            "wind_direction_10m_dominant"
        ),
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily", {})

        weather_df = pd.DataFrame({
            "Date": pd.to_datetime(daily.get("time", [])),
            "City": city,
            "temp_mean": daily.get("temperature_2m_mean", []),
            "humidity_mean": daily.get("relative_humidity_2m_mean", []),
            "precipitation": daily.get("precipitation_sum", []),
            "wind_speed_max": daily.get("wind_speed_10m_max", []),
            "wind_direction": daily.get("wind_direction_10m_dominant", []),
        })
        return weather_df

    except Exception as exc:
        print(f"[AirSense Weather Warning] Failed fetching weather for {city}: {exc}")
        return pd.DataFrame(columns=["Date", "City", "temp_mean", "humidity_mean", "precipitation", "wind_speed_max", "wind_direction"])


def get_cached_or_fetch_weather(
    cities: List[str],
    start_date: str,
    end_date: str,
    cache_path: Path = DEFAULT_WEATHER_CACHE_PATH
) -> pd.DataFrame:
    """
    Loads weather records from disk cache if present; otherwise fetches from Open-Meteo
    and saves to disk cache for reproducible offline execution.

    Args:
        cities (List[str]): List of city names.
        start_date (str): Min date.
        end_date (str): Max date.
        cache_path (Path): Path to CSV cache.

    Returns:
        pd.DataFrame: Weather dataset.
    """
    if cache_path.is_file():
        print(f"[AirSense Weather] Loading cached weather from: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        return df

    print(f"[AirSense Weather] Cache not found. Fetching from Open-Meteo API ({start_date} to {end_date})...")
    records: List[pd.DataFrame] = []
    for city in cities:
        print(f"  -> Querying Open-Meteo for {city}...")
        w_df = fetch_open_meteo_weather_for_city(city, start_date, end_date)
        if not w_df.empty:
            records.append(w_df)

    if not records:
        print("[AirSense Weather Warning] No weather data retrieved.")
        return pd.DataFrame()

    full_weather = pd.concat(records, ignore_index=True)
    full_weather.sort_values(by=["City", "Date"], inplace=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    full_weather.to_csv(cache_path, index=False)
    print(f"[AirSense Weather] Successfully cached {len(full_weather)} weather records to {cache_path}")
    return full_weather


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts calendar and cyclical features from the 'Date' column.

    Features Added:
      - day_of_week: 0 (Monday) to 6 (Sunday)
      - is_weekend: Binary indicator for traffic/commercial activity drop
      - month: 1 to 12
      - day_of_year: 1 to 366
      - season: 1 (Winter), 2 (Summer), 3 (Monsoon), 4 (Post-Monsoon)
      - Cyclical sine/cosine transformations for month and day-of-week

    Args:
        df (pd.DataFrame): Input DataFrame with 'Date' datetime.

    Returns:
        pd.DataFrame: DataFrame with temporal columns added.
    """
    data = df.copy()
    dates = data["Date"]

    data["day_of_week"] = dates.dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["day_of_month"] = dates.dt.day
    data["month"] = dates.dt.month
    data["quarter"] = dates.dt.quarter
    data["day_of_year"] = dates.dt.dayofyear

    # Cyclical encoding: ensures Dec (12) is mathematically adjacent to Jan (1)
    data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12.0)
    data["sin_day_of_week"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["cos_day_of_week"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)

    # Indian Meteorological Seasons:
    # 1: Winter (Dec, Jan, Feb)
    # 2: Summer / Pre-monsoon (Mar, Apr, May)
    # 3: Monsoon (Jun, Jul, Aug, Sep)
    # 4: Post-monsoon / Autumn (Oct, Nov)
    def map_season(month: int) -> int:
        if month in [12, 1, 2]:
            return 1
        elif month in [3, 4, 5]:
            return 2
        elif month in [6, 7, 8, 9]:
            return 3
        else:
            return 4

    data["season"] = data["month"].apply(map_season)
    return data


def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Engineers multi-day lagged features per city to capture autoregressive memory.

    Target lags: 1, 2, 3, 7 days.
    Pollutant lags: PM2.5 (1, 2, 3, 7), PM10 (1, 2, 7), NO2 (1, 7), CO (1).

    Args:
        df (pd.DataFrame): DataFrame sorted by City and Date.
        lags (Optional[List[int]]): Specific lag intervals. Defaults to [1, 2, 3, 7].

    Returns:
        pd.DataFrame: DataFrame with lagged features added.
    """
    lag_intervals = lags if lags is not None else [1, 2, 3, 7]
    data = df.copy()

    # Per-city lag generation to prevent cross-city leakage
    grouped = data.groupby("City")

    for lag in lag_intervals:
        data[f"AQI_lag_{lag}"] = grouped["AQI"].shift(lag)
        data[f"PM2.5_lag_{lag}"] = grouped["PM2.5"].shift(lag)

    # Selective lags for other key pollutants
    for lag in [1, 2, 7]:
        if "PM10" in data.columns:
            data[f"PM10_lag_{lag}"] = grouped["PM10"].shift(lag)
    for lag in [1, 7]:
        if "NO2" in data.columns:
            data[f"NO2_lag_{lag}"] = grouped["NO2"].shift(lag)
    if "CO" in data.columns:
        data["CO_lag_1"] = grouped["CO"].shift(1)

    # Trend acceleration: difference between yesterday (t-1) and day-before (t-2)
    data["AQI_diff_1_2"] = data["AQI_lag_1"] - data["AQI_lag_2"]

    return data


def add_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Engineers shifted rolling statistics (means, standard deviations, min, max).

    CRITICAL FOR NO DATA LEAKAGE:
      Rolling calculations are strictly applied on `.shift(1)` so that feature values
      at time `t` only utilize observed data up to `t-1`.

    Args:
        df (pd.DataFrame): DataFrame sorted by City and Date.
        windows (Optional[List[int]]): Rolling window sizes in days. Defaults to [3, 7, 14].

    Returns:
        pd.DataFrame: DataFrame with rolling statistical features.
    """
    window_sizes = windows if windows is not None else [3, 7, 14]
    data = df.copy()

    grouped = data.groupby("City")

    for w in window_sizes:
        # Shifted by 1 day to guarantee zero lookahead leakage
        shifted_aqi = grouped["AQI"].shift(1)
        data[f"AQI_rolling_mean_{w}"] = shifted_aqi.rolling(window=w, min_periods=w).mean()

    # 7-day rolling volatility and extremes
    shifted_7 = grouped["AQI"].shift(1)
    data["AQI_rolling_std_7"] = shifted_7.rolling(window=7, min_periods=7).std()
    data["AQI_rolling_min_7"] = shifted_7.rolling(window=7, min_periods=7).min()
    data["AQI_rolling_max_7"] = shifted_7.rolling(window=7, min_periods=7).max()

    # PM2.5 and PM10 rolling 7-day averages
    if "PM2.5" in data.columns:
        data["PM2.5_rolling_mean_7"] = grouped["PM2.5"].shift(1).rolling(7, min_periods=7).mean()
    if "PM10" in data.columns:
        data["PM10_rolling_mean_7"] = grouped["PM10"].shift(1).rolling(7, min_periods=7).mean()

    # Combustion ratio: fine particles PM2.5 to coarse PM10
    if "PM2.5_lag_1" in data.columns and "PM10_lag_1" in data.columns:
        data["PM_ratio_lag_1"] = data["PM2.5_lag_1"] / (data["PM10_lag_1"] + 1e-4)

    return data


def add_forecast_targets(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Creates lead targets for multi-horizon AQI forecasting.

    Targets:
      - target_aqi_1d: AQI at t+1 (next 24 hours)
      - target_aqi_2d: AQI at t+2 (next 48 hours)
      - target_aqi_3d: AQI at t+3 (next 72 hours)

    Args:
        df (pd.DataFrame): DataFrame sorted by City and Date.
        horizons (Optional[List[int]]): Forecast step intervals. Defaults to [1, 2, 3].

    Returns:
        pd.DataFrame: DataFrame with forecast target columns.
    """
    steps = horizons if horizons is not None else [1, 2, 3]
    data = df.copy()

    grouped = data.groupby("City")
    for step in steps:
        # Negative shift shifts future row into the current row as prediction target
        data[f"target_aqi_{step}d"] = grouped["AQI"].shift(-step)

    return data


def build_feature_dataset(
    cleaned_csv: Path | str = DEFAULT_CLEANED_PATH,
    weather_cache: Path | str = DEFAULT_WEATHER_CACHE_PATH,
    output_path: Path | str = DEFAULT_FEATURES_PATH,
    drop_unlabeled_tails: bool = True
) -> pd.DataFrame:
    """
    End-to-end feature pipeline:
      1. Loads preprocessed air quality dataset.
      2. Enriches with Open-Meteo weather records.
      3. Generates temporal calendar and cyclical features.
      4. Generates lag features (1, 2, 3, 7 days).
      5. Generates shifted rolling mean and variance features.
      6. Constructs forecast targets (1d, 2d, 3d).
      7. Prunes initialization burn-in NaNs.
      8. Persists the final ML-ready matrix to disk.

    Args:
        cleaned_csv (Path | str): Path to cleaned_city_day.csv.
        weather_cache (Path | str): Path to weather CSV cache.
        output_path (Path | str): Destination path for features.
        drop_unlabeled_tails (bool): Whether to drop tail rows where target_aqi_1d is NaN.

    Returns:
        pd.DataFrame: Ready-to-train feature DataFrame.
    """
    clean_p = Path(cleaned_csv)
    if not clean_p.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at '{clean_p}'. Run src.preprocessing first.")

    print(f"[AirSense Features] Loading cleaned data from: {clean_p}")
    df = pd.read_csv(clean_p, parse_dates=["Date"])
    df.sort_values(by=["City", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1. Fetch & Merge Open-Meteo weather
    cities = df["City"].unique().tolist()
    min_date = str(df["Date"].min().date())
    max_date = str(df["Date"].max().date())
    weather_df = get_cached_or_fetch_weather(cities, min_date, max_date, Path(weather_cache))

    if not weather_df.empty:
        print("[AirSense Features] Merging weather features with air quality records...")
        df = pd.merge(df, weather_df, on=["City", "Date"], how="left")
        # Fill any minor weather missing boundary with forward/backward fill per city
        weather_cols = ["temp_mean", "humidity_mean", "precipitation", "wind_speed_max", "wind_direction"]
        for c in weather_cols:
            if c in df.columns:
                df[c] = df.groupby("City")[c].transform(lambda s: s.ffill().bfill())

    # 2. Add temporal & calendar features
    print("[AirSense Features] Adding calendar and cyclical seasonal features...")
    df = add_temporal_features(df)

    # 3. Add lag features
    print("[AirSense Features] Engineering multi-day autoregressive lags (1, 2, 3, 7 days)...")
    df = add_lag_features(df)

    # 4. Add rolling window features
    print("[AirSense Features] Engineering shifted rolling statistics (3, 7, 14 days)...")
    df = add_rolling_features(df)

    # 5. Add forecast targets
    print("[AirSense Features] Creating forecast targets (24h, 48h, 72h ahead)...")
    df = add_forecast_targets(df)

    # 6. Drop burn-in rows caused by 14-day rolling window
    print("[AirSense Features] Dropping burn-in lag rows...")
    # Rows with NaN in AQI_rolling_mean_14 are the initial 14 days per city
    df_clean = df.dropna(subset=["AQI_rolling_mean_14"]).copy()

    if drop_unlabeled_tails:
        # Drop the last row per city where target_aqi_1d is NaN (can't train without target)
        df_clean = df_clean.dropna(subset=["target_aqi_1d"]).copy()

    df_clean.reset_index(drop=True, inplace=True)

    # 7. Save output
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_p, index=False)
    print(f"[AirSense Features] Feature engineering complete! Total samples: {len(df_clean)}, Total features: {len(df_clean.columns)}")
    print(f"[AirSense Features] Saved to: {out_p}")

    return df_clean


if __name__ == "__main__":
    features_df = build_feature_dataset()
    print("\n--- Feature Matrix Inspection ---")
    print("Columns:", list(features_df.columns))
    print("\nShape:", features_df.shape)
    print("\nSample Preview:")
    print(features_df[["City", "Date", "AQI", "AQI_lag_1", "AQI_rolling_mean_7", "temp_mean", "wind_speed_max", "target_aqi_1d"]].head())
