"""
AirSense - Streamlit Web Application
Air Quality Forecasting & Health Alert Platform for Indian Cities
"""

import sys
from pathlib import Path

# Add project root and app directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

try:
    from app.utils import (
        CITY_LOCATIONS,
        generate_city_forecasts,
        get_latest_city_status,
        load_dataset,
        load_features_dataset,
        load_models_and_columns,
    )
except (ModuleNotFoundError, ImportError):
    from utils import (
        CITY_LOCATIONS,
        generate_city_forecasts,
        get_latest_city_status,
        load_dataset,
        load_features_dataset,
        load_models_and_columns,
    )
from src.advisory import (
    CPCB_SCALE,
    calculate_outdoor_safety_score,
    get_cpcb_category,
    get_health_advisory,
)

# Page configuration
st.set_page_config(
    page_title="AirSense - Air Quality Forecasting Platform",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00B050;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .cpcb-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        color: white;
        font-size: 14px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Cache heavy resources
@st.cache_data(show_spinner=False)
def get_cached_data():
    raw_df = load_dataset()
    features_df = load_features_dataset()
    return raw_df, features_df


@st.cache_resource(show_spinner=False)
def get_cached_models():
    return load_models_and_columns()


try:
    raw_df, features_df = get_cached_data()
    best_model, multi_models, feature_cols, metrics_df = get_cached_models()
    cities_list = sorted(list(CITY_LOCATIONS.keys()))
except Exception as e:
    st.error(f"Error initializing AirSense: {e}. Please ensure Phases 1-5 have run successfully.")
    st.stop()


# Sidebar Controls
st.sidebar.image("https://img.icons8.com/clouds/200/air-quality.png", width=110)
st.sidebar.title("AirSense")
st.sidebar.caption("AI-Powered Air Quality Forecasting & Health Alert Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home & Forecast",
        "🗺️ India AQI Map",
        "📊 City Comparison",
        "🚶 'Should I Go Out?' Safety Score",
        "🔍 AI Explainability (SHAP)",
        "ℹ️ Project & Methodology",
    ],
    index=0,
)

st.sidebar.markdown("---")
selected_city = st.sidebar.selectbox("Select Target City", cities_list, index=0)
selected_group = st.sidebar.selectbox(
    "Your Demographic Profile",
    ["General Public", "Asthma Patients", "Elderly & Children", "Outdoor Workers"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔔 Early Warning Alert")
alert_threshold = st.sidebar.slider("AQI Warning Threshold", min_value=50, max_value=400, value=200, step=25)

if st.sidebar.button("🚨 Simulate / Send Alert"):
    from src.alerts import evaluate_and_dispatch_alerts, is_email_configured, is_telegram_configured
    st_status = get_latest_city_status(features_df, selected_city)
    st_fc = generate_city_forecasts(features_df, selected_city, multi_models, feature_cols)
    next_aqi = st_fc[0]["predicted_aqi"]

    outcome = evaluate_and_dispatch_alerts(
        city=selected_city,
        forecast_aqi=next_aqi,
        horizon_hours=24,
        threshold=float(alert_threshold),
        force_send=True
    )
    if is_email_configured() or is_telegram_configured():
        st.sidebar.success(f"Alert sent! Telegram: {outcome['telegram_sent']}, Email: {outcome['email_sent']}")
    else:
        st.sidebar.warning(f"Simulated alert triggered for {selected_city} (Forecast: {next_aqi:.0f} AQI). Add credentials in .env for live SMS/Email.")
    with st.sidebar.expander("View Alert Text"):
        st.code(outcome.get("message_preview", ""))

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** AirSense (College CSE Project)\n\n"
    "**ML Models:** LightGBM & XGBoost\n\n"
    "**Data:** Kaggle India AQI & Open-Meteo Weather"
)


# -----------------------------------------------------------------------------
# PAGE 1: HOME & FORECAST (24h - 72h)
# -----------------------------------------------------------------------------
if page == "🏠 Home & Forecast":
    st.title(f"🌬️ Air Quality Forecast: {selected_city}")
    st.markdown(f"*Real-time atmospheric analysis and 24 to 72-hour machine learning forecasts for **{selected_city}**.*")

    # Fetch latest status & forecasts
    status = get_latest_city_status(features_df, selected_city)
    forecasts = generate_city_forecasts(features_df, selected_city, multi_models, feature_cols)

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns([1.5, 1, 1, 1, 1])
    with c1:
        st.markdown(f"""
        <div style="background-color: {status['color']}; padding: 15px; border-radius: 10px; color: white;">
            <h4 style="margin:0; color: white;">Latest Recorded AQI</h4>
            <h1 style="margin:5px 0; color: white; font-size: 42px;">{status['aqi']}</h1>
            <span style="background: rgba(0,0,0,0.25); padding: 3px 8px; border-radius: 12px; font-weight: bold;">
                {status['category']}
            </span>
            <p style="margin: 5px 0 0 0; font-size: 11px; opacity: 0.9;">As of {status['date']}</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.metric("PM2.5", f"{status['pm25']} µg/m³")
    with c3:
        st.metric("PM10", f"{status['pm10']} µg/m³")
    with c4:
        st.metric("Temperature", f"{status['temp']} °C")
    with c5:
        st.metric("Wind Speed", f"{status['wind_speed']} km/h")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3-Day Forecast Cards
    st.subheader("📅 24 to 72-Hour Machine Learning Forecast")
    fc_cols = st.columns(3)
    for i, fc in enumerate(forecasts):
        with fc_cols[i]:
            st.markdown(f"""
            <div style="border: 2px solid {fc['color']}; border-radius: 10px; padding: 15px; background-color: #fafafa;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:bold; color: #555;">+{fc['horizon_hours']} Hours</span>
                    <span style="background-color:{fc['color']}; color:white; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:bold;">
                        {fc['category']}
                    </span>
                </div>
                <h2 style="margin:10px 0; color: #222;">{fc['predicted_aqi']} <span style="font-size:16px; color:#777;">AQI</span></h2>
                <p style="margin:0; font-size:12px; color:#666;">Date: <b>{fc['forecast_date']}</b></p>
                <p style="margin:0; font-size:12px; color:#888;">95% Confidence: {fc['lower_bound']} - {fc['upper_bound']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Interactive Forecast Plotly Chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Historical Trajectory & Forecast Horizon")

    city_recent = features_df[features_df["City"] == selected_city].sort_values("Date").tail(60)

    fig = go.Figure()

    # CPCB Category Background Shading
    fig.add_hrect(y0=0, y1=50, fillcolor="#00B050", opacity=0.1, line_width=0, annotation_text="Good (0-50)")
    fig.add_hrect(y0=51, y1=100, fillcolor="#92D050", opacity=0.1, line_width=0, annotation_text="Satisfactory (51-100)")
    fig.add_hrect(y0=101, y1=200, fillcolor="#FFC000", opacity=0.1, line_width=0, annotation_text="Moderate (101-200)")
    fig.add_hrect(y0=201, y1=300, fillcolor="#FF6600", opacity=0.1, line_width=0, annotation_text="Poor (201-300)")
    fig.add_hrect(y0=301, y1=400, fillcolor="#FF0000", opacity=0.1, line_width=0, annotation_text="Very Poor (301-400)")
    fig.add_hrect(y0=401, y1=500, fillcolor="#990033", opacity=0.1, line_width=0, annotation_text="Severe (401-500)")

    # Historical line
    fig.add_trace(go.Scatter(
        x=city_recent["Date"],
        y=city_recent["AQI"],
        name="Observed AQI",
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
    ))

    # Forecast line
    fc_dates = [pd.to_datetime(status["date"])] + [pd.to_datetime(f["forecast_date"]) for f in forecasts]
    fc_vals = [status["aqi"]] + [f["predicted_aqi"] for f in forecasts]
    fc_upper = [status["aqi"]] + [f["upper_bound"] for f in forecasts]
    fc_lower = [status["aqi"]] + [f["lower_bound"] for f in forecasts]

    # Confidence Interval Ribbon
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=fc_upper + fc_lower[::-1],
        fill="toself",
        fillcolor="rgba(255, 127, 14, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Forecast Confidence Bound",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_vals,
        name="ML Forecast Horizon",
        mode="lines+markers",
        line=dict(color="#d62728", width=3, dash="dash"),
        marker=dict(size=8, color="#d62728"),
    ))

    fig.update_layout(
        title=f"AQI Trend & Next 72 Hours Forecast for {selected_city}",
        xaxis_title="Date",
        yaxis_title="Air Quality Index (AQI)",
        yaxis=dict(range=[0, 500]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Health Advisory Card
    adv = get_health_advisory(forecasts[0]["predicted_aqi"], selected_group)
    st.subheader(f"🛡️ Tailored Health Advisory ({selected_group})")
    st.info(f"**Tomorrow's Forecast Guidance ({forecasts[0]['category']}):** {adv['advice']['summary']}")

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown(f"**🏃 Outdoor Activity:**\n{adv['advice']['activity']}")
    with ac2:
        st.markdown(f"**😷 Mask Advice:**\n{adv['advice']['mask']}")
    with ac3:
        st.markdown(f"**🏠 Indoor Air Care:**\n{adv['advice']['indoor']}")


# -----------------------------------------------------------------------------
# PAGE 2: INDIA AIR QUALITY MAP
# -----------------------------------------------------------------------------
elif page == "🗺️ India AQI Map":
    st.title("🗺️ National Air Quality Geospatial Overview")
    st.markdown("Interactive color-coded geospatial map across the 8 major metropolitan monitoring hubs.")

    # Build Map Centered on Central India
    m = folium.Map(location=[22.0, 79.0], zoom_start=5, tiles="CartoDB positron")

    city_status_list = []
    for city, coords in CITY_LOCATIONS.items():
        st_data = get_latest_city_status(features_df, city)
        city_status_list.append(st_data)

        # Popup HTML Content
        popup_html = f"""
        <div style="font-family: Arial; min-width: 160px;">
            <h4 style="margin: 0 0 5px 0; color: #333;">{city}</h4>
            <div style="background-color: {st_data['color']}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; text-align: center; margin-bottom: 8px;">
                AQI: {st_data['aqi']} ({st_data['category']})
            </div>
            <p style="margin: 2px 0; font-size: 12px;"><b>PM2.5:</b> {st_data['pm25']} µg/m³</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>PM10:</b> {st_data['pm10']} µg/m³</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Temp:</b> {st_data['temp']} °C</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Wind:</b> {st_data['wind_speed']} km/h</p>
        </div>
        """

        folium.CircleMarker(
            location=[coords["lat"], coords["lon"]],
            radius=16,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{city}: AQI {st_data['aqi']} ({st_data['category']})",
            color=st_data["color"],
            fill=True,
            fill_color=st_data["color"],
            fill_opacity=0.85,
            weight=2,
        ).add_to(m)

    st_folium(m, width="100%", height=500)

    # City Rankings Table
    st.subheader("📋 City AQI Rankings")
    rank_df = pd.DataFrame(city_status_list)[["city", "aqi", "category", "pm25", "pm10", "temp", "wind_speed"]]
    rank_df.columns = ["City", "Latest AQI", "CPCB Category", "PM2.5 (µg/m³)", "PM10 (µg/m³)", "Temp (°C)", "Wind (km/h)"]
    rank_df.sort_values(by="Latest AQI", ascending=True, inplace=True)
    rank_df.reset_index(drop=True, inplace=True)
    st.dataframe(rank_df, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 3: CITY COMPARISON
# -----------------------------------------------------------------------------
elif page == "📊 City Comparison":
    st.title("📊 Multi-City Air Quality Comparison")
    st.markdown("Compare historical trajectories, pollutant ratios, and seasonal patterns across Indian cities.")

    compare_cities = st.multiselect(
        "Select Cities to Compare",
        cities_list,
        default=["Delhi", "Mumbai", "Bengaluru"],
    )

    if not compare_cities:
        st.warning("Please select at least one city to view comparisons.")
        st.stop()

    filtered_raw = raw_df[raw_df["City"].isin(compare_cities)].sort_values("Date")

    # Time series comparison
    st.subheader("📈 Long-Term AQI Trajectories (30-Day Moving Average)")
    fig_comp = go.Figure()
    for c in compare_cities:
        c_data = filtered_raw[filtered_raw["City"] == c].copy()
        c_data["AQI_Rolling"] = c_data["AQI"].rolling(30, min_periods=7).mean()
        fig_comp.add_trace(go.Scatter(
            x=c_data["Date"],
            y=c_data["AQI_Rolling"],
            name=c,
            mode="lines",
            line=dict(width=2),
        ))

    fig_comp.update_layout(
        title="Historical AQI Trends (30-Day Moving Average)",
        xaxis_title="Date",
        yaxis_title="AQI",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Monthly Seasonality Comparison
    st.subheader("🗓️ Monthly Seasonality Patterns")
    filtered_raw["Month"] = filtered_raw["Date"].dt.month
    monthly_comp = filtered_raw.groupby(["City", "Month"])["AQI"].mean().reset_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_comp["Month_Name"] = monthly_comp["Month"].apply(lambda m: month_names[m - 1])

    fig_mon = px.line(
        monthly_comp,
        x="Month_Name",
        y="AQI",
        color="City",
        markers=True,
        title="Average AQI by Month (Winter Inversion vs Monsoon Washout)",
    )
    fig_mon.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_mon, use_container_width=True)

    # Pollutant Breakdown Boxplot
    st.subheader("🔬 Pollutant Concentration Comparison (PM2.5 & PM10)")
    p1, p2 = st.columns(2)
    with p1:
        fig_pm25 = px.box(filtered_raw, x="City", y="PM2.5", color="City", title="PM2.5 Distribution (µg/m³)")
        st.plotly_chart(fig_pm25, use_container_width=True)
    with p2:
        fig_pm10 = px.box(filtered_raw, x="City", y="PM10", color="City", title="PM10 Distribution (µg/m³)")
        st.plotly_chart(fig_pm10, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 4: "SHOULD I GO OUT?" SAFETY SCORE
# -----------------------------------------------------------------------------
elif page == "🚶 'Should I Go Out?' Safety Score":
    st.title("🚶 'Should I Go Out?' Outdoor Safety Assessment")
    st.markdown("Personalized risk intelligence assessing outdoor atmospheric suitability based on vulnerability profiles.")

    status = get_latest_city_status(features_df, selected_city)
    safety = calculate_outdoor_safety_score(status["aqi"], selected_group)

    # Big Safety Gauge / Score Card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; padding: 25px; border-left: 8px solid {safety['color']}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h4 style="margin:0; color:#555;">City: <b>{selected_city}</b> | Profile: <b>{selected_group}</b></h4>
                <h1 style="margin:10px 0; color: #222;">Safety Score: <span style="color:{safety['color']};">{safety['safety_score']} / 100</span></h1>
                <div style="background-color:{safety['color']}; color:white; padding: 6px 14px; border-radius: 20px; font-weight:bold; display:inline-block; font-size:16px;">
                    Verdict: {safety['verdict']}
                </div>
            </div>
            <div style="text-align:right;">
                <p style="margin:0; font-size:13px; color:#666;">Current AQI: <b>{status['aqi']}</b></p>
                <p style="margin:0; font-size:13px; color:#666;">Category: <b>{status['category']}</b></p>
            </div>
        </div>
        <p style="margin: 15px 0 0 0; font-size: 15px; color: #333; line-height: 1.5;">
            {safety['recommendation']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Granular Actionable Checklist
    adv = get_health_advisory(status["aqi"], selected_group)
    st.subheader("📋 Personalized Precautions Checklist")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        ### 🏃 Physical Exertion
        - **Guidance:** {adv['advice']['activity']}
        - **Morning Walks:** {'Recommended' if status['aqi'] <= 100 else 'Reschedule to afternoon or avoid'}
        - **High-Intensity Cardio:** {'Safe outdoors' if status['aqi'] <= 100 else 'Move indoors'}
        """)
    with k2:
        st.markdown(f"""
        ### 😷 Mask & PPE
        - **Recommendation:** {adv['advice']['mask']}
        - **Effective Standard:** {'None needed' if status['aqi'] <= 100 else 'Certified N95 / KN95 Respirator'}
        - **Fit Requirement:** Snug seal over nose and chin
        """)
    with k3:
        st.markdown(f"""
        ### 🏠 Home & Workplace
        - **Indoor Advice:** {adv['advice']['indoor']}
        - **Air Purifiers:** {'Optional' if status['aqi'] <= 100 else 'Activate HEPA filtration'}
        - **Ventilation:** {'Keep open' if status['aqi'] <= 100 else 'Keep windows sealed during peak traffic'}
        """)


# -----------------------------------------------------------------------------
# PAGE 5: AI EXPLAINABILITY (SHAP)
# -----------------------------------------------------------------------------
elif page == "🔍 AI Explainability (SHAP)":
    st.title("🔍 Model Explainability & Feature Drivers (SHAP)")
    st.markdown("Understand **WHY** the machine learning model makes specific AQI forecasts using game-theoretic Shapley values.")

    st.subheader("🌐 Global Feature Attribution")
    tab1, tab2, tab3 = st.tabs(["Beeswarm Summary", "Feature Importance Ranking", "Local Prediction Breakdown"])

    with tab1:
        st.markdown("**SHAP Beeswarm Plot**: Shows how higher (red) or lower (blue) values of each feature push the AQI forecast up or down.")
        beeswarm_img = MODELS_DIR / "shap_summary_beeswarm.png"
        if beeswarm_img.is_file():
            st.image(str(beeswarm_img), use_container_width=True)
        else:
            st.info("Run `python -m src.explain` to generate the beeswarm plot.")

    with tab2:
        st.markdown("**Mean Absolute SHAP Impact**: Ranks top features driving model predictions across thousands of test days.")
        bar_img = MODELS_DIR / "shap_feature_importance.png"
        if bar_img.is_file():
            st.image(str(bar_img), use_container_width=True)
        else:
            st.info("Run `python -m src.explain` to generate feature importance plots.")

    with tab3:
        st.markdown(f"**Local Case Study Waterfall**: Step-by-step breakdown for **{selected_city}**.")
        waterfall_delhi = MODELS_DIR / "shap_waterfall_delhi.png"
        waterfall_blr = MODELS_DIR / "shap_waterfall_bengaluru.png"

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.markdown("#### High Smog Episode (Delhi)")
            if waterfall_delhi.is_file():
                st.image(str(waterfall_delhi), use_container_width=True)
        with col_w2:
            st.markdown("#### Moderate Clean Day (Bengaluru)")
            if waterfall_blr.is_file():
                st.image(str(waterfall_blr), use_container_width=True)

    # Benchmark Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏆 Model Benchmarking & Accuracy")
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 6: PROJECT & METHODOLOGY
# -----------------------------------------------------------------------------
elif page == "ℹ️ Project & Methodology":
    st.title("ℹ️ Project Architecture & Viva Guide")
    st.markdown("Comprehensive overview of project methodology, algorithms, and regulatory alignment.")

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        ### 🎯 Problem Statement & Goal
        Air pollution is a major environmental and public health crisis across urban India.
        **AirSense** provides proactive 24–72h air quality forecasts, explains the meteorological
        drivers using SHAP, and delivers tailored medical advisories to vulnerable demographics.

        ### 🧠 Machine Learning Architecture
        - **Main Model:** LightGBM & XGBoost Regressors (Gradient Boosted Trees).
        - **Comparison Baseline:** Naive Persistence ($y_{t+1} = y_t$) and PyTorch LSTM.
        - **Evaluation Protocol:** Chronological time-split (no random shuffle) to prevent temporal leakage.
        - **Performance:** Achieved $R^2 \approx 0.90$, reducing RMSE by **>20%** compared to baseline.
        """)

    with m2:
        st.markdown("""
        ### 🌿 United Nations Sustainable Development Goals (SDGs)
        - **SDG 3: Good Health and Well-being:** Targeted early warnings reduce acute pulmonary emergency visits.
        - **SDG 11: Sustainable Cities and Communities:** Actionable city-level pollution insights assist urban planning.

        ### 🔬 Atmospheric Science Highlights
        - **Winter Smog:** Low temperatures cause boundary layer inversion, trapping particulates near the surface.
        - **Monsoon Scavenging:** Heavy precipitation washes out aerosols, resulting in annual air quality peaks.
        - **Particulate Dominance:** PM2.5 and PM10 drive >85% of Indian NAQI sub-index peaks.
        """)

    st.markdown("---")
    st.caption("AirSense Platform • Built with Streamlit, Plotly, Folium, LightGBM, and SHAP • 2026")
