"""
AirSense - Explainability Module (SHAP)
Author: AirSense Team
Description:
    Leverages TreeSHAP (Shapley Additive exPlanations) to interpret machine learning
    predictions globally (overall feature importance and directional impact) and
    locally (per-city, per-prediction waterfall breakdowns).
    Empowers users and examiners to understand WHY a specific AQI was forecasted.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """
    Initializes a TreeExplainer for tree ensemble models (LightGBM/XGBoost).

    TreeSHAP computes exact Shapley values in polynomial time O(TLD^2) rather than
    exponential time, providing game-theoretic fairness in credit allocation.

    Args:
        model (Any): Trained tree model instance.

    Returns:
        shap.TreeExplainer: Initialized explainer.
    """
    return shap.TreeExplainer(model)


def load_test_features_sample(
    sample_size: int = 500,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads test split records and aligns them with training feature schema.

    Args:
        sample_size (int): Number of representative samples to evaluate.
        random_state (int): Seed for sampling reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
            - X_sample (features matrix)
            - meta_sample (Date, City, actual AQI)
            - feature_names (ordered list of feature columns)
    """
    features_path = PROJECT_ROOT / "data" / "processed" / "features_city_day.csv"
    columns_path = MODELS_DIR / "feature_columns.json"

    if not features_path.is_file() or not columns_path.is_file():
        raise FileNotFoundError("Feature dataset or column definition not found.")

    with open(columns_path, "r") as f:
        feature_cols: List[str] = json.load(f)

    df = pd.read_csv(features_path, parse_dates=["Date"])
    # One-hot encode City matching training setup
    city_dummies = pd.get_dummies(df["City"], prefix="City", dtype=int)
    df_encoded = pd.concat([df, city_dummies], axis=1)

    # Use the holdout test period (2020)
    test_df = df_encoded[df_encoded["Date"] >= "2020-01-01"].copy()
    if len(test_df) > sample_size:
        sample_df = test_df.sample(n=sample_size, random_state=random_state).sort_values("Date")
    else:
        sample_df = test_df.copy()

    X_sample = sample_df[feature_cols].copy()
    meta_sample = sample_df[["Date", "City", "AQI", "target_aqi_1d"]].copy()

    return X_sample, meta_sample, feature_cols


def generate_shap_summary_plot(
    shap_explanation: shap.Explanation,
    output_path: Path | str,
    max_display: int = 15
) -> Path:
    """
    Generates and saves a SHAP beeswarm summary plot showing feature impact and direction.

    Red dots = High feature value; Blue dots = Low feature value.
    Horizontal displacement = Positive or negative push on forecasted AQI.

    Args:
        shap_explanation (shap.Explanation): Computed SHAP values.
        output_path (Path | str): File path to save PNG.
        max_display (int): Number of top features to show.

    Returns:
        Path: Path to saved figure.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 7))
    shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
    plt.title("AirSense - Global SHAP Feature Impact (Beeswarm Plot)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[AirSense SHAP] Saved Beeswarm Summary Plot to: {out}")
    return out


def generate_shap_bar_plot(
    shap_explanation: shap.Explanation,
    output_path: Path | str,
    max_display: int = 15
) -> Path:
    """
    Generates and saves a standard global SHAP feature importance bar plot (Mean |SHAP| value).

    Args:
        shap_explanation (shap.Explanation): Computed SHAP values.
        output_path (Path | str): File path to save PNG.
        max_display (int): Number of top features.

    Returns:
        Path: Path to saved figure.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_explanation, max_display=max_display, show=False)
    plt.title("AirSense - Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[AirSense SHAP] Saved Feature Importance Bar Plot to: {out}")
    return out


def generate_waterfall_plot(
    shap_explanation: shap.Explanation,
    index: int,
    output_path: Path | str,
    title_suffix: str = "",
    max_display: int = 10
) -> Path:
    """
    Generates a per-prediction waterfall plot demonstrating step-by-step
    how features moved the prediction from the base value E[f(x)] to f(x).

    Args:
        shap_explanation (shap.Explanation): Computed SHAP values.
        index (int): Row index to explain.
        output_path (Path | str): Output file path.
        title_suffix (str): Contextual label (e.g. "Delhi - Winter Smog").
        max_display (int): Number of top features.

    Returns:
        Path: Path to saved figure.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation[index], max_display=max_display, show=False)
    title = f"AirSense - Prediction Explanation (Waterfall Plot) {title_suffix}".strip()
    plt.title(title, fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[AirSense SHAP] Saved Waterfall Plot to: {out}")
    return out


def get_prediction_explanation_dict(
    shap_explanation: shap.Explanation,
    index: int,
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Extracts a structured breakdown of positive and negative drivers
    for a single prediction suitable for dynamic UI display and text summaries.

    Args:
        shap_explanation (shap.Explanation): Computed SHAP values.
        index (int): Row index.
        top_n (int): Number of positive/negative features to extract.

    Returns:
        Dict[str, Any]: Formatted explanation payload.
    """
    single_exp = shap_explanation[index]
    feature_names = single_exp.feature_names
    values = single_exp.values
    data = single_exp.data

    base_val = float(single_exp.base_values)
    pred_val = float(base_val + np.sum(values))

    # Sort contributions
    sorted_indices = np.argsort(values)

    # Negative drivers (pushing AQI down / cleaner)
    negative_drivers = []
    for idx in sorted_indices[:top_n]:
        if values[idx] < 0:
            negative_drivers.append({
                "feature": feature_names[idx],
                "actual_value": round(float(data[idx]), 2),
                "shap_impact": round(float(values[idx]), 2),
            })

    # Positive drivers (pushing AQI up / more polluted)
    positive_drivers = []
    for idx in sorted_indices[::-1][:top_n]:
        if values[idx] > 0:
            positive_drivers.append({
                "feature": feature_names[idx],
                "actual_value": round(float(data[idx]), 2),
                "shap_impact": round(float(values[idx]), 2),
            })

    return {
        "base_value": round(base_val, 2),
        "predicted_aqi": round(pred_val, 2),
        "positive_drivers": positive_drivers,
        "negative_drivers": negative_drivers,
    }


def run_explainability_pipeline() -> None:
    """
    Runs full SHAP computation on test samples and exports global and local plots.
    """
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found at {model_path}. Run src.train first.")

    print(f"[AirSense SHAP] Loading best model from: {model_path}")
    model = joblib.load(model_path)

    print("[AirSense SHAP] Loading holdout test feature records...")
    X_sample, meta_sample, feature_cols = load_test_features_sample(sample_size=400)

    print("[AirSense SHAP] Initializing TreeExplainer and calculating Shapley values...")
    explainer = get_tree_explainer(model)
    shap_values = explainer(X_sample)

    # 1. Global Beeswarm plot
    summary_path = MODELS_DIR / "shap_summary_beeswarm.png"
    generate_shap_summary_plot(shap_values, summary_path)
    generate_shap_summary_plot(shap_values, REPORTS_DIR / "shap_summary_beeswarm.png")

    # 2. Global Bar plot
    bar_path = MODELS_DIR / "shap_feature_importance.png"
    generate_shap_bar_plot(shap_values, bar_path)
    generate_shap_bar_plot(shap_values, REPORTS_DIR / "shap_feature_importance.png")

    # 3. Local Waterfall: High Pollution Sample (Delhi)
    delhi_idx = meta_sample[meta_sample["City"] == "Delhi"].index
    if not delhi_idx.empty:
        local_idx = list(meta_sample.index).index(delhi_idx[0])
        delhi_date = str(meta_sample.iloc[local_idx]["Date"].date())
        delhi_path = MODELS_DIR / "shap_waterfall_delhi.png"
        generate_waterfall_plot(shap_values, local_idx, delhi_path, title_suffix=f"(Delhi - {delhi_date})")

    # 4. Local Waterfall: Moderate/Clean Sample (Bengaluru)
    blr_idx = meta_sample[meta_sample["City"] == "Bengaluru"].index
    if not blr_idx.empty:
        local_idx = list(meta_sample.index).index(blr_idx[0])
        blr_date = str(meta_sample.iloc[local_idx]["Date"].date())
        blr_path = MODELS_DIR / "shap_waterfall_bengaluru.png"
        generate_waterfall_plot(shap_values, local_idx, blr_path, title_suffix=f"(Bengaluru - {blr_date})")

    # 5. Save pre-computed sample for Streamlit instant interactive widget
    joblib.dump({
        "X_sample": X_sample.head(50),
        "meta_sample": meta_sample.head(50),
        "feature_names": feature_cols,
        "base_value": float(explainer.expected_value),
    }, MODELS_DIR / "shap_interactive_cache.joblib")

    print("[AirSense SHAP] Explainability pipeline completed successfully!")


if __name__ == "__main__":
    run_explainability_pipeline()
