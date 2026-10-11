# 🎓 AirSense: Viva Voce & Technical Defense Guide
## Top 25 College CSE Viva Questions & Expert Answers

---

### **Section 1: Machine Learning & Modeling**

#### Q1: What machine learning model did you use, and which one performed best?
> **Answer:** We trained and benchmarked four distinct approaches:
> 1. **Naive Persistence Baseline ($y_{t+1} = y_t$)** ($MAE = 18.42$, $RMSE = 33.99$, $R^2 = 0.8378$)
> 2. **XGBoost Regressor** ($MAE = 15.70$, $RMSE = 26.90$, $R^2 = 0.8984$)
> 3. **LightGBM Regressor** ($MAE = 15.66$, $RMSE = 27.00$, $R^2 = 0.8977$)
> 4. **PyTorch LSTM** ($MAE = 19.87$, $RMSE = 31.68$, $R^2 = 0.8591$)
>
> **LightGBM** achieved the lowest Mean Absolute Error ($15.66$) with an $R^2$ close to $0.90$, achieving a **20.56% reduction in RMSE** over the naive baseline.

#### Q2: Why did a tree-based ensemble (LightGBM) outperform deep learning (LSTM)?
> **Answer:** Deep learning sequence models like LSTMs excel on raw, continuous, high-frequency signals (e.g., audio, raw seconds/minutes sensor feeds) where millions of sequential steps exist. 
> In our daily aggregated tabular dataset (12,000+ rows across 8 cities), explicit feature engineering (1, 2, 3, 7-day lags, shifted moving averages, and non-linear weather interactions) transforms the problem into a tabular regression task. Gradient-boosted decision trees construct orthogonal decision splits on tabular data more sample-efficiently and without vanishing/exploding gradients.

#### Q3: What is data leakage in time-series forecasting, and how did you prevent it?
> **Answer:** Data leakage occurs when future information contaminates the model training process, yielding artificially inflated test accuracy that fails in production. We prevented it through two strict design choices:
> 1. **Chronological Splitting:** We used a time-based split (Train: 2015 to June 2019, Validation: July to Dec 2019, Test: Jan to June 2020) rather than random K-Fold cross-validation.
> 2. **Shifted Rolling Windows:** All rolling statistics (means, standard deviations) were calculated strictly on `.shift(1)` data, ensuring that features at day $t$ only contain information observed up to day $t-1$.

#### Q4: Why is a Naive Persistence Baseline necessary?
> **Answer:** In time-series forecasting, atmospheric air quality has high natural autocorrelation (tomorrow is often similar to today). A model that claims 85% accuracy might simply be memorizing today's value. Comparing against $y_{t+1} = y_t$ proves whether our feature engineering and machine learning algorithm actually add genuine predictive intelligence. Our models beat persistence by over **20% in RMSE**.

#### Q5: What evaluation metrics did you use, and why?
> **Answer:**
> - **MAE (Mean Absolute Error):** Measures average absolute deviation in AQI points; robust to extreme outliers and easily understood by non-technical stakeholders.
> - **RMSE (Root Mean Squared Error):** Penalizes large forecasting errors quadratically, which is critical in public health where missing a hazardous smog spike is dangerous.
> - **$R^2$ Score (Coefficient of Determination):** Quantifies the percentage of variance explained by the model (~90%).
> - **MAPE (Mean Absolute Percentage Error):** Provides scale-independent relative percentage error (~13.2%).

---

### **Section 2: Data Engineering & Preprocessing**

#### Q6: How did you handle missing sensor values?
> **Answer:**
> 1. **Pre-monitoring Trimming:** Certain stations (like Mumbai and Kolkata) were only commissioned in 2018. We trimmed unmonitored leading void dates rather than backfilling 3 years of non-existent data with artificial flatlines.
> 2. **Time-Weighted Linear Interpolation:** Short sensor dropouts (1 to 7 days) were bridged along the chronological timeline, preserving the physical atmospheric trajectory unlike global mean imputation.
> 3. **Forward/Backward Fill:** Handled boundary edges per city.

#### Q7: Why did you cap AQI at 500 instead of using standard 3-sigma outlier deletion?
> **Answer:**
> 1. In India, extreme pollution events (Diwali fireworks, Punjab/Haryana stubble burning in November) are **genuine atmospheric phenomena**, not false data. Standard 3-sigma deletion would delete the very hazardous spikes we are trying to predict.
> 2. Deleting rows creates broken calendar gaps, which destroys daily autoregressive lag sequences ($t-1, t-7$).
> 3. Under CPCB NAQI regulatory guidelines, 500 is the defined ceiling (*Severe*). Capping uncalibrated sensor drift (e.g. Ahmedabad >2000) at 500 preserves the true emergency hazard level while preventing gradient explosion during tree splitting.

#### Q8: Where did the meteorological data come from?
> **Answer:** From the **Open-Meteo Historical Archive and Forecast REST API**. It is an open, non-commercial API that provides daily temperature, relative humidity, precipitation sum, and wind speed by latitude and longitude coordinates without requiring an API key. We cached over 16,000 daily weather records locally to enable offline reproducibility.

#### Q9: What is cyclical harmonic encoding, and why is it used?
> **Answer:** Traditional integer representations of months (1 to 12) or days of the week (0 to 6) imply that December (12) is far from January (1). By transforming dates using sine and cosine functions:
> $$\sin\_month = \sin\left(\frac{2\pi \cdot m}{12}\right), \quad \cos\_month = \cos\left(\frac{2\pi \cdot m}{12}\right)$$
> we map dates onto a 2D trigonometric circle where December 31 and January 1 are mathematically adjacent.

---

### **Section 3: Explainable AI (SHAP)**

#### Q10: What is SHAP, and why did you choose it over feature importance?
> **Answer:** Standard Gini/gain feature importance in tree models shows *how much* a feature was used, but does not reveal **directionality** (whether a higher value increases or decreases the prediction) and suffers from inconsistency bias.  
> **SHAP (Shapley Additive exPlanations)** is rooted in cooperative game theory. It computes the exact marginal contribution of each feature across all possible subsets, satisfying mathematical properties of local accuracy, missingness, and consistency.

#### Q11: What were the most important features discovered by SHAP?
> **Answer:**
> 1. **Immediate Particulate Inertia:** Current day $AQI$ and $PM_{2.5}$ carry the highest positive weight ($\approx 44.8$ and $15.6$ points).
> 2. **Wind Speed:** `wind_speed_max` displays strong **negative SHAP values**—high wind speeds physically disperse stagnant pollutants, lowering the forecasted AQI.
> 3. **Precipitation:** `precipitation` acts as a negative contributor due to atmospheric wet scavenging (rain washout).
> 4. **Combustion Proxies:** $CO$ and $NO_x$ feature prominently during winter morning and evening commute windows.

#### Q12: What is the difference between a SHAP Beeswarm plot and a Waterfall plot?
> **Answer:**
> - **Beeswarm Plot (Global):** Shows the overall impact of all features across hundreds of test samples, where dot color represents feature value (red = high, blue = low) and horizontal displacement shows whether it increased or decreased AQI.
> - **Waterfall Plot (Local):** Explains a **single specific prediction** (e.g. Delhi on Dec 15), starting from the base value $E[f(x)]$ and adding/subtracting individual feature contributions step-by-step to arrive at the final forecasted value $f(x)$.

---

### **Section 4: Health Advisory, Alerts, & Full-Stack**

#### Q13: How is the Indian CPCB AQI calculated?
> **Answer:** CPCB calculates individual sub-indices for up to 8 pollutants ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3, NH_3, Pb$) using piece-wise linear breakpoint functions. The overall AQI is the **maximum** of these sub-indices, provided that at least 3 pollutants are monitored (one of which must be $PM_{2.5}$ or $PM_{10}$). In India, the maximum sub-index is almost always driven by particulate matter.

#### Q14: How does the "Should I Go Out?" safety score work?
> **Answer:** It converts numerical AQI into a normalized 0–100 scale using piece-wise CPCB thresholds (e.g. Good = 90–100, Severe = 0–10). It then applies a **demographic vulnerability penalty multiplier**:
> - General Public: $1.00\times$
> - Outdoor Workers: $1.15\times$
> - Elderly & Children: $1.25\times$
> - Asthma / Respiratory Patients: $1.35\times$  
> This ensures that on a Moderate day (AQI = 160), a healthy person receives "Generally Safe" while an asthmatic receives "Caution Advised - Keep Inhaler Accessible".

#### Q15: How does the automated alerting system operate?
> **Answer:** Implemented in `src/alerts.py`:
> 1. Compares forecasted AQI against a configurable hazard threshold (default: 200).
> 2. If crossed, compiles demographic health precautions and dispatches notifications via:
>    - **SMTP TLS Email:** Using Python's standard `smtplib` with responsive HTML templates.
>    - **Telegram Bot API:** Transmitting instant push notifications via Telegram HTTP webhook.
> 3. All secrets are loaded strictly from `.env` via `python-dotenv`.

#### Q16: How is CI/CD and MLOps implemented in AirSense?
> **Answer:** We configured a **GitHub Actions workflow** (`.github/workflows/retrain_and_refresh.yml`) that runs on a scheduled cron (`0 0 * * *`, daily at midnight UTC / 05:30 AM IST). It automatically sets up Python 3.11, validates data, pulls updated Open-Meteo weather data, retrains forecasting models, recalculates SHAP figures, checks alert thresholds, and commits updated model artifacts back to the repository.

#### Q17: Which United Nations Sustainable Development Goals does AirSense address?
> **Answer:**
> - **SDG 3 (Good Health and Well-being) - Target 3.9:** Substantially reducing mortality and morbidity from hazardous air pollution through predictive, personalized early warnings.
> - **SDG 11 (Sustainable Cities and Communities) - Target 11.6:** Reducing the adverse environmental impact of cities by improving public awareness and municipal forecasting.

---

### **Section 5: Live Demonstration Script**

1. **Step 1 (Home & Forecast):** Show the 8-city dropdown, highlight Delhi's current AQI, and walk through the **24h, 48h, 72h forecast cards** with confidence intervals.
2. **Step 2 (Plotly Forecast Chart):** Show the CPCB colored horizontal bands (*Good to Severe*) and the shaded 95% confidence ribbon.
3. **Step 3 (Geospatial Map):** Switch to the **India AQI Map** tab; click markers on Mumbai and Bengaluru to display pollutant breakdown popups.
4. **Step 4 (City Comparison):** Compare Delhi vs. Bengaluru to highlight the stark winter inversion contrast.
5. **Step 5 (Safety Score):** Toggle the user profile from *General Public* to *Asthma Patients* and show how the safety score drops from 60 to 46 with tailored inhaler precautions.
6. **Step 6 (SHAP Explainability):** Explain the **Beeswarm plot** and highlight how **wind speed** acts as a negative driver (dispersion).
7. **Step 7 (Alert Trigger):** Click **"🚨 Simulate / Send Alert"** in the sidebar to prove real-time warning generation.
