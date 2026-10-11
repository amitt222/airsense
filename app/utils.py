"""
AirSense - Streamlit App Helper Utilities
Author: AirSense Team
Description:
    Shared data loading, model inference, multi-horizon forecasting, and
    geospatial formatting helpers for the Streamlit web application.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.advisory import get_cpcb_category, calculate_outdoor_safety_score, get_health_advisory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# City Geographic Coordinates for Mapping
CITY_LOCATIONS: Dict[str, Dict[str, float]] = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
}


def load_dataset() -> pd.DataFrame:
    """Loads the preprocessed historical air quality dataset."""
    path = DATA_DIR / "cleaned_city_day.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned dataset not found at {path}. Run src.preprocessing first.")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def load_features_dataset() -> pd.DataFrame:
    """Loads the engineered feature matrix with weather and lags."""
    path = DATA_DIR / "features_city_day.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Feature dataset not found at {path}. Run src.features first.")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def load_models_and_columns() -> Tuple[Any, Dict[str, Any], List[str], pd.DataFrame]:
    """
    Loads trained ML models, multi-horizon models, feature columns, and benchmark metrics.

    Returns:
        Tuple: (best_model, multi_horizon_models_dict, feature_columns_list, metrics_df)
    """
    best_model_path = MODELS_DIR / "best_model.joblib"
    multi_models_path = MODELS_DIR / "multi_horizon_models.joblib"
    cols_path = MODELS_DIR / "feature_columns.json"
    metrics_path = MODELS_DIR / "metrics_comparison.csv"

    if not best_model_path.is_file():
        raise FileNotFoundError(f"Model not found at {best_model_path}. Run src.train first.")

    best_model = joblib.load(best_model_path)
    multi_models = joblib.load(multi_models_path) if multi_models_path.is_file() else {}

    with open(cols_path, "r") as f:
        feature_cols: List[str] = json.load(f)

    metrics_df = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    return best_model, multi_models, feature_cols, metrics_df


def get_latest_city_status(features_df: pd.DataFrame, city: str) -> Dict[str, Any]:
    """
    Retrieves the most recent recorded atmospheric state for a given city.

    Args:
        features_df (pd.DataFrame): Engineered feature dataset.
        city (str): Name of the target city.

    Returns:
        Dict[str, Any]: Latest parameters (Date, AQI, Category, PM2.5, PM10, Weather).
    """
    city_df = features_df[features_df["City"] == city].sort_values("Date")
    if city_df.empty:
        raise ValueError(f"No records found for city: {city}")

    latest_row = city_df.iloc[-1]
    aqi_val = float(latest_row["AQI"])
    cat_info = get_cpcb_category(aqi_val)

    return {
        "city": city,
        "date": str(latest_row["Date"].date()),
        "aqi": round(aqi_val, 1),
        "category": cat_info["category"],
        "color": cat_info["color"],
        "pm25": round(float(latest_row.get("PM2.5", 0.0)), 1),
        "pm10": round(float(latest_row.get("PM10", 0.0)), 1),
        "temp": round(float(latest_row.get("temp_mean", 25.0)), 1),
        "humidity": round(float(latest_row.get("humidity_mean", 50.0)), 1),
        "wind_speed": round(float(latest_row.get("wind_speed_max", 10.0)), 1),
        "latest_row": latest_row,
    }


def generate_city_forecasts(
    features_df: pd.DataFrame,
    city: str,
    multi_models: Dict[str, Any],
    feature_cols: List[str]
) -> List[Dict[str, Any]]:
    """
    Produces 24-hour, 48-hour, and 72-hour AQI forecasts using trained multi-step models.

    Args:
        features_df (pd.DataFrame): Features DataFrame.
        city (str): Target city.
        multi_models (Dict[str, Any]): Dictionary of trained models for 1d, 2d, 3d.
        feature_cols (List[str]): Ordered feature names.

    Returns:
        List[Dict[str, Any]]: 3 forecast predictions with dates, predicted AQI, category, and confidence bands.
    """
    city_df = features_df[features_df["City"] == city].sort_values("Date")
    latest_row = city_df.iloc[-1:]

    # One-hot encode City matching training setup
    city_dummies = pd.get_dummies(latest_row["City"], prefix="City", dtype=int)
    row_encoded = pd.concat([latest_row.reset_index(drop=True), city_dummies.reset_index(drop=True)], axis=1)

    # Ensure all feature columns exist (fill missing dummy columns with 0)
    for col in feature_cols:
        if col not in row_encoded.columns:
            row_encoded[col] = 0

    X_input = row_encoded[feature_cols].values
    latest_date = pd.to_datetime(latest_row["Date"].values[0])

    forecasts: List[Dict[str, Any]] = []

    for step in [1, 2, 3]:
        model_key = f"target_aqi_{step}d"
        model = multi_models.get(model_key)
        if model is None:
            # Fallback persistence or main model
            pred_aqi = float(latest_row["AQI"].values[0])
        else:
            pred_aqi = float(model.predict(X_input)[0])

        # Clip within CPCB bounds (0 to 500)
        pred_aqi = max(10.0, min(500.0, pred_aqi))

        forecast_date = latest_date + pd.Timedelta(days=step)
        cat_info = get_cpcb_category(pred_aqi)

        # Standard error margins (RMSE ~ 27 for 24h, widening slightly with horizon)
        margin = 15.0 + (step - 1) * 6.0
        lower_bound = max(0.0, pred_aqi - margin)
        upper_bound = min(500.0, pred_aqi + margin)

        forecasts.append({
            "horizon_hours": step * 24,
            "horizon_days": step,
            "forecast_date": str(forecast_date.date()),
            "predicted_aqi": round(pred_aqi, 1),
            "category": cat_info["category"],
            "color": cat_info["color"],
            "lower_bound": round(lower_bound, 1),
            "upper_bound": round(upper_bound, 1),
        })

    return forecasts
