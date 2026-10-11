"""
AirSense - EDA and Visualizations Generator
Generates and saves publication-quality exploratory plots to `notebooks/plots/`.
Covers missing data diagnostics, outlier treatment, per-city trends,
seasonality, and pollutant correlation matrices.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' is importable when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import filter_target_cities, load_raw_city_day
from src.preprocessing import clean_and_preprocess_dataset, save_cleaned_data

# Output directory for static plots
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def plot_missing_values(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Plots missing value percentage comparison before and after cleaning."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    raw_missing = (raw_df.isna().mean() * 100).sort_values(ascending=False)
    cleaned_missing = (cleaned_df.isna().mean() * 100).reindex(raw_missing.index).fillna(0)

    sns.barplot(x=raw_missing.values, y=raw_missing.index, hue=raw_missing.index, ax=ax1, palette="Reds_r", legend=False)
    ax1.set_title("Missing Values Before Cleaning (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Percentage Missing (%)")

    sns.barplot(x=cleaned_missing.values, y=cleaned_missing.index, hue=cleaned_missing.index, ax=ax2, palette="Greens_r", legend=False)
    ax2.set_title("Missing Values After Cleaning (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Percentage Missing (%)")

    plt.tight_layout()
    output_path = PLOTS_DIR / "01_missing_values_profile.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[AirSense EDA] Saved: {output_path}")


def plot_city_distributions(df: pd.DataFrame) -> None:
    """Boxplot of AQI distribution across target cities."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Order by median AQI
    city_order = df.groupby("City")["AQI"].median().sort_values(ascending=False).index

    sns.boxplot(
        data=df,
        x="City",
        y="AQI",
        hue="City",
        order=city_order,
        palette="Spectral",
        ax=ax,
        legend=False,
        showfliers=True,
    )
    # Highlight CPCB Thresholds
    ax.axhline(50, color="green", linestyle="--", alpha=0.7, label="Good (50)")
    ax.axhline(100, color="lightgreen", linestyle="--", alpha=0.7, label="Satisfactory (100)")
    ax.axhline(200, color="orange", linestyle="--", alpha=0.7, label="Moderate (200)")
    ax.axhline(300, color="red", linestyle="--", alpha=0.7, label="Poor (300)")
    ax.axhline(400, color="darkred", linestyle="--", alpha=0.7, label="Very Poor (400)")

    ax.set_title("AQI Distribution by City (Ordered by Median AQI)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Air Quality Index (AQI)")
    ax.set_xlabel("Target City")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    output_path = PLOTS_DIR / "02_city_aqi_distributions.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[AirSense EDA] Saved: {output_path}")


def plot_historical_trends(df: pd.DataFrame) -> None:
    """Plots 30-day rolling average AQI historical trends across major cities."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Focus on representative cities across India for visual clarity
    spotlight_cities = ["Delhi", "Mumbai", "Bengaluru", "Kolkata"]
    palette = {"Delhi": "#d62728", "Mumbai": "#1f77b4", "Bengaluru": "#2ca02c", "Kolkata": "#9467bd"}

    for city in spotlight_cities:
        city_sub = df[df["City"] == city].sort_values("Date").copy()
        city_sub["AQI_Rolling30"] = city_sub["AQI"].rolling(window=30, min_periods=7).mean()
        ax.plot(city_sub["Date"], city_sub["AQI_Rolling30"], label=city, color=palette[city], linewidth=2)

    # Highlight COVID-19 Lockdown window (March 2020)
    ax.axvspan(pd.to_datetime("2020-03-24"), pd.to_datetime("2020-05-31"), color="gold", alpha=0.3, label="2020 National Lockdown")

    ax.set_title("Historical AQI Trajectories (30-Day Rolling Mean) & Lockdown Impact", fontsize=13, fontweight="bold")
    ax.set_ylabel("AQI (30-Day Moving Average)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")

    plt.tight_layout()
    output_path = PLOTS_DIR / "03_historical_trends_lockdown.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[AirSense EDA] Saved: {output_path}")


def plot_seasonality(df: pd.DataFrame) -> None:
    """Plots monthly average AQI demonstrating winter vs monsoon seasonality."""
    df_copy = df.copy()
    df_copy["Month"] = df_copy["Date"].dt.month
    monthly_avg = df_copy.groupby(["City", "Month"])["AQI"].mean().reset_index()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=monthly_avg,
        x="Month",
        y="AQI",
        hue="City",
        marker="o",
        linewidth=2,
        ax=ax,
    )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_title("Monthly AQI Seasonality Across Indian Cities (Winter Peaks vs Monsoon Dips)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average AQI")
    ax.set_xlabel("Month of the Year")

    # Annotate winter inversion and monsoon washout
    ax.annotate("Winter Smog Peak\n(Inversion & Low Wind)", xy=(11, 280), xytext=(8.5, 340),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6))
    ax.annotate("Monsoon Washout\n(Precipitation)", xy=(7, 70), xytext=(4, 40),
                arrowprops=dict(facecolor="blue", shrink=0.05, width=1, headwidth=6))

    plt.tight_layout()
    output_path = PLOTS_DIR / "04_monthly_seasonality.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[AirSense EDA] Saved: {output_path}")


def plot_correlations(df: pd.DataFrame) -> None:
    """Correlation heatmap between pollutants and AQI."""
    pollutants = ["PM2.5", "PM10", "NO2", "NOx", "CO", "SO2", "O3", "AQI"]
    corr = df[pollutants].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Pollutants vs Final AQI", fontsize=13, fontweight="bold")

    plt.tight_layout()
    output_path = PLOTS_DIR / "05_pollutant_correlations.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[AirSense EDA] Saved: {output_path}")


def plot_cpcb_categories(df: pd.DataFrame) -> None:
    """Bar plot of CPCB AQI Bucket proportion per city."""
    category_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    colors = ["#2ecc71", "#a8e05f", "#f39c12", "#e67e22", "#e74c3c", "#8e44ad"]
    color_map = dict(zip(category_order, colors))

    grouped = (
        df.groupby(["City", "AQI_Bucket"])
        .size()
        .unstack(fill_value=0)
    )
    grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
    grouped = grouped.reindex(columns=category_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    grouped.plot(
        kind="bar",
        stacked=True,
        color=[color_map[c] for c in category_order],
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_title("Proportion of Days in Each CPCB AQI Category by City", fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage of Days (%)")
    ax.set_xlabel("City")
    ax.legend(title="CPCB Category", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    output_path = PLOTS_DIR / "06_cpcb_category_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[AirSense EDA] Saved: {output_path}")


def run_full_eda() -> None:
    """Executes the full EDA routine and saves all figures."""
    print("[AirSense EDA] Loading and filtering raw dataset...")
    raw_df = load_raw_city_day()
    target_raw_df = filter_target_cities(raw_df)

    print("[AirSense EDA] Executing cleaning pipeline...")
    cleaned_df = clean_and_preprocess_dataset(target_raw_df)
    save_cleaned_data(cleaned_df)

    print("[AirSense EDA] Generating Figure 1: Missing values profile...")
    plot_missing_values(target_raw_df, cleaned_df)

    print("[AirSense EDA] Generating Figure 2: City AQI distributions...")
    plot_city_distributions(cleaned_df)

    print("[AirSense EDA] Generating Figure 3: Historical trends & lockdown...")
    plot_historical_trends(cleaned_df)

    print("[AirSense EDA] Generating Figure 4: Monthly seasonality...")
    plot_seasonality(cleaned_df)

    print("[AirSense EDA] Generating Figure 5: Correlation matrix...")
    plot_correlations(cleaned_df)

    print("[AirSense EDA] Generating Figure 6: CPCB Category distribution...")
    plot_cpcb_categories(cleaned_df)

    print("\n[AirSense EDA] All EDA plots generated successfully in notebooks/plots/!")


if __name__ == "__main__":
    run_full_eda()
