"""
AirSense - Model Evaluation Module
Author: AirSense Team
Description:
    Computes rigorous evaluation metrics (MAE, RMSE, R2, MAPE) for time-series forecasting.
    Includes baseline comparison routines and formats benchmark results into
    clean Markdown and CSV tables for presentation and viva explanations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes core regression metrics on test sets.

    Args:
        y_true (np.ndarray | pd.Series): Ground truth target AQI values.
        y_pred (np.ndarray | pd.Series): Model predicted AQI values.
        model_name (str): Label for the model.

    Returns:
        Dict[str, Any]: Dictionary containing MAE, RMSE, R2, and MAPE.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    # Clean any NaN pairs if present
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    yt = yt[mask]
    yp = yp[mask]

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))

    # Safe MAPE calculation avoiding division by zero
    non_zero_mask = yt > 0
    if non_zero_mask.any():
        mape = float(np.mean(np.abs((yt[non_zero_mask] - yp[non_zero_mask]) / yt[non_zero_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE (%)": round(mape, 2),
    }


def compare_models(
    results: List[Dict[str, Any]],
    baseline_model_name: str = "Naive Persistence (t-1)"
) -> pd.DataFrame:
    """
    Generates a consolidated comparison table and calculates percentage improvement
    in MAE and RMSE relative to the naive baseline.

    Args:
        results (List[Dict[str, Any]]): List of metric dictionaries from `calculate_metrics`.
        baseline_model_name (str): Name of the baseline model for improvement benchmark.

    Returns:
        pd.DataFrame: Formatted comparison table sorted by lowest MAE.
    """
    df = pd.DataFrame(results)

    # Locate baseline row
    baseline_row = df[df["Model"] == baseline_model_name]
    if not baseline_row.empty:
        base_mae = baseline_row["MAE"].values[0]
        base_rmse = baseline_row["RMSE"].values[0]

        df["MAE Improvement (%)"] = df["MAE"].apply(
            lambda x: round(((base_mae - x) / base_mae) * 100.0, 2)
        )
        df["RMSE Improvement (%)"] = df["RMSE"].apply(
            lambda x: round(((base_rmse - x) / base_rmse) * 100.0, 2)
        )
    else:
        df["MAE Improvement (%)"] = 0.0
        df["RMSE Improvement (%)"] = 0.0

    # Sort models by ascending MAE (best performing at top)
    df.sort_values(by="MAE", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def print_markdown_comparison_table(comparison_df: pd.DataFrame) -> str:
    """
    Formats comparison DataFrame into a clean Markdown table.

    Args:
        comparison_df (pd.DataFrame): Comparison DataFrame.

    Returns:
        str: Markdown formatted table string.
    """
    md_str = comparison_df.to_markdown(index=False)
    print("\n" + "=" * 80)
    print("                     AIRSENSE: MODEL BENCHMARK COMPARISON")
    print("=" * 80)
    print(md_str)
    print("=" * 80 + "\n")
    return md_str


if __name__ == "__main__":
    # Unit demonstration with synthetic benchmark
    mock_true = np.array([120, 150, 180, 210, 250])
    mock_naive = np.array([115, 140, 170, 195, 230])
    mock_xgb = np.array([122, 148, 178, 208, 246])

    r_naive = calculate_metrics(mock_true, mock_naive, "Naive Persistence (t-1)")
    r_xgb = calculate_metrics(mock_true, mock_xgb, "XGBoost Regressor")

    table = compare_models([r_naive, r_xgb])
    print_markdown_comparison_table(table)
