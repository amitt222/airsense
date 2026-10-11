"""
AirSense - Model Training & Benchmarking Pipeline
Author: AirSense Team
Description:
    Implements chronological (time-based) train/validation/test splitting,
    trains Naive Persistence, XGBoost, LightGBM, and an LSTM / tree comparison model.
    Evaluates models using MAE and RMSE against the baseline, logs results to a
    comparison table, and saves the best performing production models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from src.evaluate import calculate_metrics, compare_models, print_markdown_comparison_table

# Check optional PyTorch availability for LSTM
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features_city_day.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Non-feature metadata columns to exclude from training matrix
EXCLUDE_COLUMNS = [
    "Date",
    "City",
    "AQI_Bucket",
    "target_aqi_1d",
    "target_aqi_2d",
    "target_aqi_3d",
]


def load_and_encode_dataset(
    features_csv: Path | str = FEATURES_CSV
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads the engineered feature matrix and applies One-Hot Encoding to the City column.

    Args:
        features_csv (Path | str): Path to features_city_day.csv.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Encoded DataFrame and list of feature column names.
    """
    path = Path(features_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Feature dataset not found at '{path}'. Run src.features first.")

    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values(by=["City", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # One-hot encode City to avoid imposing false ordinal ranking
    city_dummies = pd.get_dummies(df["City"], prefix="City", drop_first=False, dtype=int)
    df_encoded = pd.concat([df, city_dummies], axis=1)

    feature_cols = [
        col for col in df_encoded.columns
        if col not in EXCLUDE_COLUMNS
    ]
    return df_encoded, feature_cols


def create_time_based_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_end: str = "2019-06-30",
    val_end: str = "2019-12-31"
) -> Dict[str, Any]:
    """
    Splits dataset chronologically into Train, Validation, and Test sets.

    Temporal Windows:
      - Train Set: Inception to 2019-06-30 (Long-term pattern learning)
      - Validation Set: 2019-07-01 to 2019-12-31 (Hyperparameter tuning & early stopping)
      - Test Set: 2020-01-01 to 2020-07-01 (Out-of-sample evaluation including 2020 lockdown)

    Why Time-Based Split?
      Standard random k-fold / shuffling mixes future records into training and past records
      into testing. This leaks autoregressive signals and yields artificially inflated test scores.
      A chronological split strictly measures real-world forward generalization.

    Args:
        df (pd.DataFrame): Encoded dataset.
        feature_cols (List[str]): List of predictor feature names.
        train_end (str): Cutoff date for training.
        val_end (str): Cutoff date for validation.

    Returns:
        Dict[str, Any]: Dict containing split matrices and metadata.
    """
    train_mask = df["Date"] <= train_end
    val_mask = (df["Date"] > train_end) & (df["Date"] <= val_end)
    test_mask = df["Date"] > val_end

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    split_data = {
        "X_train": train_df[feature_cols].values,
        "y_train_1d": train_df["target_aqi_1d"].values,
        "y_train_2d": train_df["target_aqi_2d"].values,
        "y_train_3d": train_df["target_aqi_3d"].values,

        "X_val": val_df[feature_cols].values,
        "y_val_1d": val_df["target_aqi_1d"].values,
        "y_val_2d": val_df["target_aqi_2d"].values,
        "y_val_3d": val_df["target_aqi_3d"].values,

        "X_test": test_df[feature_cols].values,
        "y_test_1d": test_df["target_aqi_1d"].values,
        "y_test_2d": test_df["target_aqi_2d"].values,
        "y_test_3d": test_df["target_aqi_3d"].values,

        "test_meta": test_df[["Date", "City", "AQI", "target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]].copy(),
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
    }

    print(f"[AirSense Split] Chronological Data Split:")
    print(f"  -> Train samples : {len(train_df)} ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  -> Val samples   : {len(val_df)} ({val_df['Date'].min().date()} to {val_df['Date'].max().date()})")
    print(f"  -> Test samples  : {len(test_df)} ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    return split_data


def run_naive_baseline(split_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the standard Naive Persistence baseline (Tomorrow's AQI = Today's AQI).

    Formula: y_hat(t+1) = y(t)

    Args:
        split_data (Dict[str, Any]): Split data dictionary.

    Returns:
        Dict[str, Any]: Baseline metrics and test predictions.
    """
    # In test_meta, 'AQI' is today's recorded AQI
    y_test = split_data["y_test_1d"]
    y_pred_naive = split_data["test_meta"]["AQI"].values

    metrics = calculate_metrics(y_test, y_pred_naive, model_name="Naive Persistence (t-1)")
    return {"metrics": metrics, "predictions": y_pred_naive}


def train_xgboost_regressor(split_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains an XGBoost Regressor with early stopping on the validation set.

    Args:
        split_data (Dict[str, Any]): Split data dictionary.

    Returns:
        Dict[str, Any]: Trained model, predictions, and metrics.
    """
    X_train, y_train = split_data["X_train"], split_data["y_train_1d"]
    X_val, y_val = split_data["X_val"], split_data["y_val_1d"]
    X_test, y_test = split_data["X_test"], split_data["y_test_1d"]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, model_name="XGBoost Regressor")
    return {"model": model, "predictions": y_pred, "metrics": metrics}


def train_lightgbm_regressor(split_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains a LightGBM Regressor with early stopping on the validation set.

    Args:
        split_data (Dict[str, Any]): Split data dictionary.

    Returns:
        Dict[str, Any]: Trained model, predictions, and metrics.
    """
    X_train, y_train = split_data["X_train"], split_data["y_train_1d"]
    X_val, y_val = split_data["X_val"], split_data["y_val_1d"]
    X_test, y_test = split_data["X_test"], split_data["y_test_1d"]

    model = lgb.LGBMRegressor(
        n_estimators=400,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, model_name="LightGBM Regressor")
    return {"model": model, "predictions": y_pred, "metrics": metrics}


class SimpleLSTM(nn.Module):
    """PyTorch Lightweight LSTM Sequence Regressor for Time-Series Comparison."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Take the output of the last time step
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


def train_lstm_regressor(split_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains a PyTorch LSTM comparison model (or fallback Random Forest if PyTorch is unavailable).

    Args:
        split_data (Dict[str, Any]): Split data dictionary.

    Returns:
        Dict[str, Any]: Model dictionary with metrics and predictions.
    """
    if not HAS_TORCH:
        print("[AirSense Model] PyTorch not detected. Training Random Forest Regressor as comparison model...")
        rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(split_data["X_train"], split_data["y_train_1d"])
        y_pred = rf.predict(split_data["X_test"])
        metrics = calculate_metrics(split_data["y_test_1d"], y_pred, model_name="Random Forest (Comparison)")
        return {"model": rf, "predictions": y_pred, "metrics": metrics}

    print("[AirSense Model] Training PyTorch LSTM Comparison Model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(split_data["X_train"])
    X_test_scaled = scaler.transform(split_data["X_test"])

    # Reshape into sequence format (batch_size, sequence_length=1, input_features)
    X_train_seq = torch.tensor(X_train_scaled[:, np.newaxis, :], dtype=torch.float32)
    y_train_t = torch.tensor(split_data["y_train_1d"], dtype=torch.float32)
    X_test_seq = torch.tensor(X_test_scaled[:, np.newaxis, :], dtype=torch.float32)
    y_test = split_data["y_test_1d"]

    dataset = TensorDataset(X_train_seq, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    input_dim = X_train_scaled.shape[1]
    model = SimpleLSTM(input_dim=input_dim, hidden_dim=48, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # Train for 20 epochs
    model.train()
    for epoch in range(20):
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_seq).numpy()

    metrics = calculate_metrics(y_test, y_pred, model_name="PyTorch LSTM (Comparison)")
    return {"model": model, "predictions": y_pred, "metrics": metrics, "scaler": scaler}


def train_multi_horizon_models(
    split_data: Dict[str, Any],
    best_algorithm: str = "xgboost"
) -> Dict[str, Any]:
    """
    Trains dedicated models for 24-hour (1d), 48-hour (2d), and 72-hour (3d) ahead forecasts.

    Args:
        split_data (Dict[str, Any]): Split data dictionary.
        best_algorithm (str): 'xgboost' or 'lightgbm'.

    Returns:
        Dict[str, Any]: Dictionary containing models for each horizon.
    """
    horizon_models = {}
    print("\n[AirSense Model] Training Multi-Horizon Forecasting Models (24h, 48h, 72h)...")

    for horizon in [1, 2, 3]:
        y_train = split_data[f"y_train_{horizon}d"]
        y_val = split_data[f"y_val_{horizon}d"]

        # Drop any edge NaNs in multi-step targets
        valid_train = ~np.isnan(y_train)
        valid_val = ~np.isnan(y_val)

        X_tr = split_data["X_train"][valid_train]
        y_tr = y_train[valid_train]
        X_v = split_data["X_val"][valid_val]
        y_v = y_val[valid_val]

        if best_algorithm == "xgboost":
            m = xgb.XGBRegressor(
                n_estimators=350,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        else:
            m = lgb.LGBMRegressor(
                n_estimators=350,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(30, verbose=False)])

        horizon_models[f"target_aqi_{horizon}d"] = m
        print(f"  -> Horizon +{horizon*24} Hours ({horizon} Day{'s' if horizon>1 else ''}) model trained successfully.")

    return horizon_models


def run_training_pipeline() -> Tuple[pd.DataFrame, Any]:
    """
    Orchestrates full training, benchmarking, comparison table creation, and artifact saving.

    Returns:
        Tuple[pd.DataFrame, Any]: Comparison DataFrame and best model.
    """
    print("[AirSense Training] Loading feature dataset...")
    df_encoded, feature_cols = load_and_encode_dataset()

    print("[AirSense Training] Splitting chronologically...")
    split_data = create_time_based_split(df_encoded, feature_cols)

    # 1. Baseline
    print("\n--- Model 1: Naive Persistence Baseline ---")
    naive_res = run_naive_baseline(split_data)
    print("Naive Persistence:", naive_res["metrics"])

    # 2. XGBoost
    print("\n--- Model 2: XGBoost Regressor ---")
    xgb_res = train_xgboost_regressor(split_data)
    print("XGBoost Regressor:", xgb_res["metrics"])

    # 3. LightGBM
    print("\n--- Model 3: LightGBM Regressor ---")
    lgb_res = train_lightgbm_regressor(split_data)
    print("LightGBM Regressor:", lgb_res["metrics"])

    # 4. Comparison Model (LSTM / RF)
    print("\n--- Model 4: Comparison Model ---")
    comp_res = train_lstm_regressor(split_data)
    print("Comparison Model:", comp_res["metrics"])

    # 5. Model Benchmarking Table
    all_metrics = [
        naive_res["metrics"],
        xgb_res["metrics"],
        lgb_res["metrics"],
        comp_res["metrics"],
    ]
    comparison_table = compare_models(all_metrics, baseline_model_name="Naive Persistence (t-1)")
    print_markdown_comparison_table(comparison_table)

    # Determine Best Model (lowest MAE)
    best_model_name = comparison_table.iloc[0]["Model"]
    print(f"[AirSense Training] Best Performing Model: {best_model_name}")

    if "XGBoost" in best_model_name:
        best_model = xgb_res["model"]
        best_algo = "xgboost"
        best_preds = xgb_res["predictions"]
    elif "LightGBM" in best_model_name:
        best_model = lgb_res["model"]
        best_algo = "lightgbm"
        best_preds = lgb_res["predictions"]
    else:
        best_model = xgb_res["model"]
        best_algo = "xgboost"
        best_preds = xgb_res["predictions"]

    # 6. Train Multi-Horizon Models (1d, 2d, 3d) using best architecture
    multi_models = train_multi_horizon_models(split_data, best_algorithm=best_algo)

    # 7. Persist Artifacts
    print("\n[AirSense Training] Saving artifacts to models/ directory...")
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(xgb_res["model"], MODELS_DIR / "xgboost_model.joblib")
    joblib.dump(lgb_res["model"], MODELS_DIR / "lightgbm_model.joblib")
    joblib.dump(multi_models, MODELS_DIR / "multi_horizon_models.joblib")

    # Save feature names
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save comparison metrics
    comparison_table.to_csv(MODELS_DIR / "metrics_comparison.csv", index=False)
    with open(MODELS_DIR / "metrics_comparison.json", "w") as f:
        json.dump(comparison_table.to_dict(orient="records"), f, indent=2)

    # Save test set evaluation predictions with metadata for explainability & charts
    test_results_df = split_data["test_meta"].copy()
    test_results_df["pred_naive"] = naive_res["predictions"]
    test_results_df["pred_xgboost"] = xgb_res["predictions"]
    test_results_df["pred_lightgbm"] = lgb_res["predictions"]
    test_results_df["pred_comparison"] = comp_res["predictions"]
    test_results_df["pred_best"] = best_preds
    test_results_df.to_csv(MODELS_DIR / "test_evaluation_results.csv", index=False)

    print(f"[AirSense Training] All artifacts successfully saved to {MODELS_DIR}!")
    return comparison_table, best_model


if __name__ == "__main__":
    run_training_pipeline()
