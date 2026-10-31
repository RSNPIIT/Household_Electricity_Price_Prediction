"""
Experiments & Capstone Evaluation Module for Household Electricity Consumption Prediction.
Executes:
- Experiments A, B, C, D (Feature Ablation Study)
- Experiment E (Capstone Forecast Horizon Investigation: 10m, 30m, 1h, 6h, 24h)
- Generates Expected Graphs 10 through 14:
  10. Actual vs Predicted Consumption
  11. Residual Distribution
  12. Forecast-Horizon vs RMSE
  13. Feature Importance
  14. Model Comparison
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

from src.models_scratch import (
    LinearRegressionScratch,
    RidgeRegressionScratch,
    LassoRegressionScratch,
    DecisionTreeRegressorScratch,
    RandomForestRegressorScratch,
    ExtraTreesRegressorScratch,
    calculate_metrics
)
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb


def get_model_suite(subsample_trees: bool = True) -> Dict[str, Any]:
    """
    Instantiate the required suite of ML algorithms:
    - 6 From-Scratch Algorithms (Linear, Ridge, Lasso, DecisionTree, RandomForest, ExtraTrees)
    - Gradient Boosting Benchmark (HistGradientBoosting / LightGBM)
    """
    n_est = 35 if subsample_trees else 60
    depth = 10 if subsample_trees else 14
    
    models = {
        'Linear Regression (Scratch)': LinearRegressionScratch(fit_intercept=True),
        'Ridge Regression (Scratch)': RidgeRegressionScratch(alpha=10.0, fit_intercept=True),
        'Lasso Regression (Scratch)': LassoRegressionScratch(alpha=0.01, max_iter=300, fit_intercept=True),
        'Decision Tree (Scratch)': DecisionTreeRegressorScratch(max_depth=depth, min_samples_split=30, min_samples_leaf=15),
        'Random Forest (Scratch)': RandomForestRegressorScratch(n_estimators=n_est, max_depth=depth, min_samples_split=30, min_samples_leaf=15),
        'Extra Trees (Scratch)': ExtraTreesRegressorScratch(n_estimators=n_est, max_depth=depth, min_samples_split=30, min_samples_leaf=15),
        'Gradient Boosting (GBDT)': HistGradientBoostingRegressor(max_iter=80, max_depth=8, learning_rate=0.1, random_state=42)
    }
    return models


# =============================================================================
# REMAINING EXPECTED GRAPHS (10 TO 14)
# =============================================================================
def plot_actual_vs_predicted(
    y_test: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    timestamps: Optional[pd.DatetimeIndex] = None,
    save_path: str = "figures/fig10_actual_vs_predicted.png",
    sample_window: int = 288 # 48 hours of 10-minute resampled or 288 minutes
):
    """
    Graph 10: Actual vs. Predicted consumption over a representative test slice.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 6))
    
    yt = np.asarray(y_test)[:sample_window]
    x_axis = timestamps[:sample_window] if timestamps is not None else np.arange(sample_window)
    
    plt.plot(x_axis, yt, color='#0f172a', linewidth=2.2, label='Actual Consumption (Truth)')
    
    colors = {
        'Linear Regression (Scratch)': '#94a3b8',
        'Ridge Regression (Scratch)': '#64748b',
        'Lasso Regression (Scratch)': '#38bdf8',
        'Decision Tree (Scratch)': '#f59e0b',
        'Random Forest (Scratch)': '#059669',
        'Extra Trees (Scratch)': '#10b981',
        'Gradient Boosting (GBDT)': '#dc2626'
    }
    
    # Plot top models for clean visual clarity
    for name, yp in predictions_dict.items():
        if name in ['Random Forest (Scratch)', 'Gradient Boosting (GBDT)', 'Linear Regression (Scratch)']:
            yp_slice = np.asarray(yp)[:sample_window]
            c = colors.get(name, '#6366f1')
            ls = '--' if 'Gradient' in name else '-.'
            plt.plot(x_axis, yp_slice, color=c, linestyle=ls, linewidth=1.7, label=name)
            
    plt.title("Actual vs. Predicted Global Active Power (Test Slice)", pad=14, weight='bold')
    plt.xlabel("Timeline" if timestamps is not None else "Time Steps (Minutes)")
    plt.ylabel("Global Active Power (kW)")
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 10: {save_path}")


def plot_residual_distribution(
    y_test: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: str = "figures/fig11_residual_distribution.png"
):
    """
    Graph 11: Residual distribution (Error histogram & KDE).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rf_name = 'Random Forest (Scratch)'
    gb_name = 'Gradient Boosting (GBDT)'
    
    model_to_plot = gb_name if gb_name in predictions_dict else list(predictions_dict.keys())[0]
    residuals = np.asarray(y_test) - np.asarray(predictions_dict[model_to_plot])
    
    # Subplot 1: Residual Histogram & KDE
    sns.histplot(residuals, bins=70, kde=True, ax=axes[0], color='#2563eb', stat="density", alpha=0.6)
    axes[0].axvline(0, color='#dc2626', linestyle='--', linewidth=1.5, label='Zero Error Benchmark')
    axes[0].axvline(np.mean(residuals), color='#059669', linestyle='-', linewidth=1.5,
                    label=f'Mean Error: {np.mean(residuals):.3f} kW')
    axes[0].set_title(f"Residual Distribution ({model_to_plot})", weight='bold')
    axes[0].set_xlabel("Residual (Actual - Predicted) [kW]")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True, facecolor='white')
    
    # Subplot 2: Residuals vs Predicted (Homoscedasticity Check)
    preds = np.asarray(predictions_dict[model_to_plot])
    sample_sub = np.random.choice(len(preds), min(5000, len(preds)), replace=False)
    axes[1].scatter(preds[sample_sub], residuals[sample_sub], alpha=0.3, color='#4338ca', s=12)
    axes[1].axhline(0, color='#dc2626', linestyle='--', linewidth=1.5)
    axes[1].set_title(f"Residuals vs. Predicted Values ({model_to_plot})", weight='bold')
    axes[1].set_xlabel("Predicted Global Active Power (kW)")
    axes[1].set_ylabel("Residual (kW)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 11: {save_path}")


def plot_forecast_horizon_vs_rmse(
    horizon_results_df: pd.DataFrame,
    save_path: str = "figures/fig12_forecast_horizon_vs_rmse.png"
):
    """
    Graph 12: Forecast Horizon vs. RMSE (Capstone Investigation).
    Displays error growth across 10m, 30m, 1h, 6h, 24h.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5.5))
    
    for model_name in horizon_results_df['Model'].unique():
        sub = horizon_results_df[horizon_results_df['Model'] == model_name]
        plt.plot(sub['Horizon_Label'], sub['RMSE'], marker='o', linewidth=2.0, label=model_name)
        
    plt.title("Capstone Investigation: Prediction Degradation vs. Forecasting Horizon", pad=14, weight='bold')
    plt.xlabel("Forecasting Horizon")
    plt.ylabel("Root Mean Squared Error (RMSE) [kW]")
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 12: {save_path}")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "Random Forest (Scratch)",
    top_n: int = 15,
    save_path: str = "figures/fig13_feature_importance.png"
):
    """
    Graph 13: Top Feature Importances (Mean Decrease in Impurity / Tree Splits).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Reverse order so top feature is at the top
    imp_df = imp_df.iloc[::-1]
    
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='#0284c7', edgecolor='#0369a1', height=0.65)
    plt.title(f"Top {top_n} Most Important Features — {model_name}", pad=14, weight='bold')
    plt.xlabel("Relative Feature Importance (MDI / Gain)")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 13: {save_path}")


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = "figures/fig14_model_comparison.png"
):
    """
    Graph 14: Model Comparison (Grouped bar chart across MAE, RMSE, MAPE).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [('RMSE', 'RMSE (kW) [Lower is Better]', '#2563eb'),
               ('MAE', 'MAE (kW) [Lower is Better]', '#059669'),
               ('MAPE', 'MAPE (%) [Lower is Better]', '#d97706')]
               
    for ax, (metric, label, col) in zip(axes, metrics):
        sorted_df = comparison_df.sort_values(metric)
        sns.barplot(data=sorted_df, x=metric, y='Model', ax=ax, color=col, edgecolor='black', alpha=0.85)
        ax.set_title(f"Model Comparison: {metric}", weight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel("")
        
        # Annotate values
        for p in ax.patches:
            val = p.get_width()
            ax.annotate(f"{val:.3f}", (val, p.get_y() + p.get_height() / 2.),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
                        
    plt.suptitle("Overall Benchmark of From-Scratch Algorithms vs. GBDT", y=1.02, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 14: {save_path}")
