# Household Electricity Consumption Prediction Using Traditional Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: From--Scratch](https://img.shields.io/badge/Algorithms-From--Scratch-brightgreen.svg)](#from-scratch-algorithm-implementations)
[![Dataset: UCI-2.07M](https://img.shields.io/badge/Dataset-UCI%202.07M%20Rows-orange.svg)](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

An end-to-end, high-performance time-series regression system for predicting individual household electricity consumption using historical electrical measurements and temporal information. This repository contains complete **from-scratch implementations of all non-GBDT algorithms**, temporal exploratory data analysis, causal feature engineering, chronological validation, capstone forecast-horizon investigations, and error analysis.

---

## 1. Problem Statement & Real-World Motivation

Electricity providers and modern smart-home energy management systems (HEMS) require precise, short-to-medium-term forecasts of electrical power consumption. Accurate load forecasting enables:
1. **Demand Planning & Grid Dispatching**: Minimizing spinning reserve costs and preventing grid overloads during sudden consumption surges.
2. **Dynamic Energy Management**: Scheduling domestic high-load appliances (water heaters, electric vehicles, HVAC) during off-peak tariff periods.
3. **Anomaly & Fault Detection**: Spotting abnormal consumption spikes indicative of equipment malfunction, electrical leakage, or unauthorized tapping.

The core regression objective is to forecast future **Global Active Power ($y_{t+h}$)** across various forecasting horizons ($h \in \{1\text{m}, 10\text{m}, 30\text{m}, 1\text{h}, 6\text{h}, 24\text{h}\}$) using strictly historical measurements available at prediction time $t$.

---

## 2. Dataset Overview & Preprocessing Pipeline

- **Source**: [UCI Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Observations**: $2,075,259$ one-minute continuous timestamps spanning nearly 4 years (December 16, 2006, 17:24 to November 26, 2010, 21:02).
- **Sampling Frequency**: $1\text{ observation / minute}$.
- **Missing Value Handling**: Exactly $25,979$ timestamps ($1.25\%$) had missing electrical measurements (`?`). For time-series data, **forward-fill (`ffill`)** was applied to preserve temporal causality and prevent future information leakage.
- **Derived Physical Metric**: Remainder active sub-metering representing unmonitored household electrical equipment:
  $$\text{Sub\_metering\_remainder} = \max\left(0, \frac{\text{Global\_active\_power} \times 1000}{60} - \sum_{i=1}^3 \text{Sub\_metering}_i\right)$$

---

## 3. From-Scratch Algorithm Implementations

As required by the project specifications, all classical machine learning regression algorithms are implemented **from mathematical first principles** in `src/models_scratch.py`:

| Algorithm | Implementation Mechanism | Mathematical Formulation |
| :--- | :--- | :--- |
| **Linear Regression** | Ordinary Least Squares via SVD Pseudoinverse | $\hat{\beta} = (X^T X)^{-1} X^T y = V \Sigma^+ U^T y$ |
| **Ridge Regression** | $L_2$-Regularized Analytical Normal Equations | $\hat{\beta}_{\text{ridge}} = (X_c^T X_c + \alpha I)^{-1} X_c^T y_c$ |
| **Lasso Regression** | $L_1$-Regularized Cyclic Coordinate Descent | $\beta_j \leftarrow \frac{S(\rho_j, \alpha)}{z_j}$, where $S(z, \gamma) = \text{sign}(z)\max(0, |z| - \gamma)$ |
| **Decision Tree Regressor** | Recursive MSE Variance Reduction with Quantiles | $\Delta \text{MSE} = \text{Var}(S) - \left[\frac{\|S_L\|}{\|S\|}\text{Var}(S_L) + \frac{\|S_R\|}{\|S\|}\text{Var}(S_R)\right]$ |
| **Random Forest Regressor** | Bootstrap Bagging + Random Feature Subspace ($m_{\text{try}}$) | $\hat{y}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x), \quad m_{\text{try}} = \lfloor\sqrt{p}\rfloor$ |
| **Extra Trees Regressor** | Extremely Randomized Trees with Uniform Cut-Points | Draws random cut-points $c_j \sim \mathcal{U}[\min(X_j), \max(X_j)]$ across features |
| **Gradient Boosting (GBDT)**| HistGradientBoosting / LightGBM Benchmark | $\hat{y}_M(x) = \sum_{m=1}^M \eta \cdot h_m(x)$, fitting trees to negative gradients |

---

## 4. Chronological Validation & Leakage Prevention

> [!IMPORTANT]
> **Zero Shuffling Requirement**: Traditional cross-validation or random train-test splitting introduces catastrophic future data leakage into historical time-series.
> Data is split chronologically into:
> - **Training Set**: First $70\%$ ($2006\text{--}2009$)
> - **Validation Set**: Middle $15\%$ ($2009\text{--}2010$)
> - **Test Set**: Final $15\%$ ($2010$)
>
> All rolling window features ($10\text{m}, 30\text{m}, 60\text{m}$) are strictly computed on $y_{t-1}$ to ensure the current observation $y_t$ is never leaked into input predictors.

---

## 5. Experimental Results & Ablation Study

### 5.1 Experiment Ablation Benchmark (Feature Increments)

| Experiment | Features Added | Model | RMSE (kW) | MAE (kW) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Exp A: Baseline** | 7 Electrical Variables | Linear Regression (Scratch)<br>Decision Tree (Scratch)<br>Gradient Boosting (GBDT) | 0.1916<br>0.2055<br>0.1926 | 0.0726<br>0.0922<br>0.0804 | 9.39%<br>12.09%<br>9.64% | 0.9406<br>0.9316<br>0.9399 |
| **Exp B: Temporal** | +14 Calendar & Cyclical ($\sin/\cos$) | Linear Regression (Scratch)<br>Decision Tree (Scratch)<br>Gradient Boosting (GBDT) | 0.2087<br>0.2111<br>0.1963 | 0.1198<br>0.0967<br>0.0840 | 17.40%<br>11.91%<br>9.98% | 0.9295<br>0.9279<br>0.9376 |
| **Exp C: Lags** | +11 Autoregressive Lags ($1, 10, 30, 60, 1440\text{m}$) | Linear Regression (Scratch)<br>Decision Tree (Scratch)<br>Gradient Boosting (GBDT) | 0.2006<br>0.2098<br>**0.1746** | 0.1164<br>0.0945<br>**0.0726** | 17.41%<br>11.69%<br>**8.96%** | 0.9289<br>0.9222<br>**0.9461** |
| **Exp D: Rolling Stats** | +12 Rolling Stats (Mean, Std, Min, Max) | Linear Regression (Scratch)<br>Decision Tree (Scratch)<br>Gradient Boosting (GBDT) | 0.1938<br>0.2059<br>**0.1707** | 0.1021<br>0.0935<br>**0.0720** | 14.55%<br>11.95%<br>**9.36%** | 0.9336<br>0.9251<br>**0.9485** |

### 5.2 Full Model Comparison Benchmark (Feature Set D)

| Algorithm | Implementation Type | RMSE (kW) | MAE (kW) | MAPE (%) | $R^2$ Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lasso Regression** | **From Scratch** | **0.1835** | **0.0710** | **9.26%** | **0.9405** | 0.08s |
| **Ridge Regression** | **From Scratch** | 0.1935 | 0.1013 | 14.40% | 0.9338 | 0.002s |
| **Linear Regression** | **From Scratch** | 0.1938 | 0.1021 | 14.55% | 0.9336 | 0.03s |
| **Random Forest** | **From Scratch** | 0.1969 | 0.0836 | 10.51% | 0.9315 | 3.96s |
| **Extra Trees** | **From Scratch** | 0.2017 | 0.0868 | 11.01% | 0.9281 | 2.78s |
| **Decision Tree** | **From Scratch** | 0.2059 | 0.0935 | 11.95% | 0.9251 | 2.51s |
| **Gradient Boosting** | GBDT Benchmark | **0.1707** | **0.0720** | **9.36%** | **0.9485** | 0.46s |

---

## 6. Capstone Investigation: Forecasting Horizons ($E_1 \to E_5$)

How does prediction performance behave as the forecasting horizon increases?

| Horizon Label | Ahead Steps | Target | Random Forest (Scratch) RMSE | GBDT Benchmark RMSE | Linear Regression RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **E1 (Next 10 Minutes)** | $t + 10\text{m}$ | $y_{t+10}$ | **0.4668 kW** | 0.4693 kW | 0.6781 kW |
| **E2 (Next 30 Minutes)** | $t + 30\text{m}$ | $y_{t+30}$ | **0.6061 kW** | 0.6516 kW | 1.1509 kW |
| **E3 (Next 1 Hour)** | $t + 60\text{m}$ | $y_{t+60}$ | **0.7089 kW** | 0.7579 kW | 1.8652 kW |
| **E4 (Next 6 Hours)** | $t + 360\text{m}$ | $y_{t+360}$ | **0.8470 kW** | 0.9360 kW | 2.6729 kW |
| **E5 (Next 24 Hours)** | $t + 1440\text{m}$ | $y_{t+1440}$ | **0.7924 kW** | 0.8551 kW | 2.2426 kW |

### Key Capstone Insights:
1. **Short-Term Autoregressive Dominance**: Within $10\text{--}30$ minutes, immediate electrical state and recent rolling means deliver tight accuracy ($\text{RMSE} < 0.6\text{ kW}$).
2. **Linear Degradation vs. Non-linear Resilience**: Linear regression errors explode beyond 1 hour because linear models cannot capture phase-shifted non-linear diurnal cycles. Tree ensembles retain meaningful predictive power.
3. **Diurnal Cycle Recovery at 24 Hours ($E_5$)**: Interestingly, RMSE improves from $0.847\text{ kW}$ at $6\text{h}$ to $0.792\text{ kW}$ at $24\text{h}$. This occurs because household routines exhibit high 24-hour periodicity ($t - 1440\text{m}$ lag aligns with the exact time-of-day).

---

## 7. The 14 Expected Figures

All figures are automatically generated in publication-quality high resolution ($300\text{ DPI}$) inside `figures/`:

1. `fig01_active_power_distribution.png`: Histogram & KDE highlighting right-skewed standby power vs. heavy appliance peaks.
2. `fig02_power_consumption_time_series.png`: 4-year macro view with daily average and 30-day rolling trendlines.
3. `fig03_hourly_consumption_pattern.png`: 24-hour diurnal profile demonstrating morning (8 AM) and evening (8 PM) peaks.
4. `fig04_daily_consumption_pattern.png`: Day-of-month aggregate consumption across days 1–31.
5. `fig05_weekly_consumption_pattern.png`: Boxplot comparison showing elevated daytime consumption on weekends.
6. `fig06_monthly_consumption_pattern.png`: Seasonal dynamics demonstrating elevated winter heating vs. summer troughs.
7. `fig07_correlation_heatmap.png`: Multivariate Pearson correlation matrix between electrical features and sub-meterings.
8. `fig08_lag_correlation_acf_pacf.png`: Autocorrelation (ACF) decay across 60 minute lags with $95\%$ confidence bounds.
9. `fig09_rolling_mean_plot.png`: Smoothing comparison contrasting raw 1-minute noise with 1-hour and 24-hour rolling averages.
10. `fig10_actual_vs_predicted.png`: Actual vs. Predicted time-series tracking over test set.
11. `fig11_residual_distribution.png`: Error distribution and homoscedasticity residual-vs-predicted scatter.
12. `fig12_forecast_horizon_vs_rmse.png`: Error growth curves across horizons $E_1 \to E_5$.
13. `fig13_feature_importance.png`: Top 15 ranked features via Mean Decrease in Impurity.
14. `fig14_model_comparison.png`: Three-panel comparative bar chart across RMSE, MAE, and MAPE.

---

## 8. Repository Structure

```text
Household_Electricity_Price_Prediction/
├── dataset.txt                        # UCI dataset (tracked via Git LFS)
├── .gitattributes                     # Git LFS configuration
├── LICENSE                            # MIT Open Source License
├── figures/                           # 14 publication-ready figures
│   ├── fig01_active_power_distribution.png
│   ├── fig02_power_consumption_time_series.png
│   ├── fig03_hourly_consumption_pattern.png
│   ├── fig04_daily_consumption_pattern.png
│   ├── fig05_weekly_consumption_pattern.png
│   ├── fig06_monthly_consumption_pattern.png
│   ├── fig07_correlation_heatmap.png
│   ├── fig08_lag_correlation_acf_pacf.png
│   ├── fig09_rolling_mean_plot.png
│   ├── fig10_actual_vs_predicted.png
│   ├── fig11_residual_distribution.png
│   ├── fig12_forecast_horizon_vs_rmse.png
│   ├── fig13_feature_importance.png
│   └── fig14_model_comparison.png
├── results/                           # Experimental metrics & horizon logs
│   ├── experiment_ablation_results.csv
│   ├── forecast_horizon_results.csv
│   └── model_comparison_summary.csv
├── src/                               # Modular Python source code
│   ├── __init__.py
│   ├── preprocessing.py               # Data loading, missing value imputation, split
│   ├── feature_engineering.py         # Temporal, cyclical, lag, and rolling features
│   ├── models_scratch.py              # 6 algorithms implemented from scratch
│   ├── eda.py                         # Plotting functions for EDA figures 1-9
│   └── experiments.py                 # Horizon evaluation & figures 10-14
├── run_pipeline.py                    # End-to-end command-line execution script
├── Household_Electricity_Consumption_Prediction.ipynb  # Interactive annotated notebook
└── README.md                          # Comprehensive project documentation
```

---

## 9. Quickstart & Reproducibility Guide

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/RSNPIIT/Household_Electricity_Price_Prediction.git
cd Household_Electricity_Price_Prediction
pip install -r requirements.txt
```

### Pull Dataset via Git LFS
```bash
git lfs pull
```

### Run End-to-End Pipeline
To run the full preprocessing, EDA, ablation experiments, and horizon investigation:
```bash
python run_pipeline.py --dataset dataset.txt --figures_dir figures --results_dir results
```
All 14 figures will be updated in `figures/` and metrics saved in `results/`.
