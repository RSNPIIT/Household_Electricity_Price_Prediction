"""
End-to-End Orchestrator Pipeline for Household Electricity Consumption Prediction.
Runs:
1. Ingestion & Preprocessing
2. Temporal EDA (Generates Figures 1 - 9)
3. Feature Engineering & Chronological Train/Val/Test Splitting
4. Experiments A, B, C, D (Feature Ablation Benchmark)
5. Experiment E (Capstone Forecast Horizon Investigation: 10m, 30m, 1h, 6h, 24h)
6. Model Comparison & Error Analysis (Generates Figures 10 - 14)
7. Exports results to CSV and figures to figures/
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_raw_dataset, audit_missing_values, handle_missing_values, chronological_split
from src.feature_engineering import build_experiment_features
from src.eda import generate_all_eda_plots
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
from src.experiments import (
    get_model_suite,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_forecast_horizon_vs_rmse,
    plot_feature_importance,
    plot_model_comparison
)


def run_pipeline(
    dataset_path: str = "dataset.txt",
    figures_dir: str = "figures",
    results_dir: str = "results",
    train_sample_size: int = 60000,
    test_sample_size: int = 20000
):
    start_total = time.time()
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*80)
    print("PROJECT 5: HOUSEHOLD ELECTRICITY CONSUMPTION PREDICTION")
    print("TRADITIONAL MACHINE LEARNING FROM SCRATCH & TIME-SERIES REGRESSION")
    print("="*80)

    # -------------------------------------------------------------------------
    # STEP 1: INGESTION & AUDIT
    # -------------------------------------------------------------------------
    df_raw = load_raw_dataset(dataset_path)
    audit = audit_missing_values(df_raw)
    print("\n[*] Initial Missing Value Audit:")
    print(audit.to_string())

    # -------------------------------------------------------------------------
    # STEP 2: IMPUTATION & PREPROCESSING
    # -------------------------------------------------------------------------
    df_clean = handle_missing_values(df_raw, method='ffill')

    # -------------------------------------------------------------------------
    # STEP 3: TEMPORAL EDA & EXPECTED GRAPHS (1 TO 9)
    # -------------------------------------------------------------------------
    generate_all_eda_plots(df_clean, figures_dir=figures_dir)

    # -------------------------------------------------------------------------
    # STEP 4: EXPERIMENTS A, B, C, D (FEATURE ABLATION STUDY)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: EXECUTING FEATURE ABLATION EXPERIMENTS (A, B, C, D)")
    print("="*80)
    ablation_experiments = {
        'Exp A (Baseline: Electrical Only)': 'A',
        'Exp B (Temporal & Cyclical)': 'B',
        'Exp C (Autoregressive Lags)': 'C',
        'Exp D (Rolling Statistics - Full)': 'D'
    }

    ablation_records = []
    
    # We test the impact of feature engineering using Linear Regression & Random Forest
    for exp_label, exp_code in ablation_experiments.items():
        print(f"\n---> Building Features for {exp_label}...")
        X, y, feat_names = build_experiment_features(
            df_clean, experiment_type=exp_code, target_col='Global_active_power', horizon_minutes=1
        )
        print(f"     Feature count: {len(feat_names)} | Samples: {len(X):,}")
        
        # Chronological split (70% Train, 15% Val, 15% Test)
        n = len(X)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
        
        # Subsample for high-speed training of ensemble models
        if train_sample_size and len(X_train) > train_sample_size:
            # Take the most recent train samples chronologically
            X_tr_sub = X_train.iloc[-train_sample_size:].values
            y_tr_sub = y_train.iloc[-train_sample_size:].values
        else:
            X_tr_sub = X_train.values
            y_tr_sub = y_train.values

        if test_sample_size and len(X_test) > test_sample_size:
            X_te_sub = X_test.iloc[:test_sample_size].values
            y_te_sub = y_test.iloc[:test_sample_size].values
        else:
            X_te_sub = X_test.values
            y_te_sub = y_test.values

        models_ablation = get_model_suite(subsample_trees=True)
        # Test key benchmarks on this ablation
        for m_name in ['Linear Regression (Scratch)', 'Decision Tree (Scratch)', 'Gradient Boosting (GBDT)']:
            model = models_ablation[m_name]
            t0 = time.time()
            model.fit(X_tr_sub, y_tr_sub)
            fit_time = time.time() - t0
            
            preds = model.predict(X_te_sub)
            metrics = calculate_metrics(y_te_sub, preds)
            
            print(f"     {m_name:28s} | RMSE: {metrics['RMSE']:.4f} kW | MAE: {metrics['MAE']:.4f} kW | R2: {metrics['R2']:.4f} ({fit_time:.2f}s)")
            ablation_records.append({
                'Experiment': exp_label,
                'Model': m_name,
                'Num_Features': len(feat_names),
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'R2': metrics['R2'],
                'Fit_Time_s': fit_time
            })

    ablation_df = pd.DataFrame(ablation_records)
    ablation_csv_path = os.path.join(results_dir, "experiment_ablation_results.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"\n[+] Saved Feature Ablation Results to: {ablation_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 5: FULL MODEL BENCHMARK (ON FULL FEATURE SET D)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: COMPREHENSIVE BENCHMARK OF ALL 7 ALGORITHMS (FULL FEATURE SET D)")
    print("="*80)

    X_full, y_full, full_feat_names = build_experiment_features(
        df_clean, experiment_type='D', target_col='Global_active_power', horizon_minutes=1
    )
    
    n = len(X_full)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    X_train_full = X_full.iloc[:train_end]
    y_train_full = y_full.iloc[:train_end]
    X_test_full = X_full.iloc[val_end:]
    y_test_full = y_full.iloc[val_end:]

    X_tr = X_train_full.iloc[-train_sample_size:].values if train_sample_size else X_train_full.values
    y_tr = y_train_full.iloc[-train_sample_size:].values if train_sample_size else y_train_full.values

    test_slice_len = min(test_sample_size, len(X_test_full))
    X_te = X_test_full.iloc[:test_slice_len].values
    y_te = y_test_full.iloc[:test_slice_len].values
    test_timestamps = X_test_full.index[:test_slice_len]

    full_suite = get_model_suite(subsample_trees=True)
    comparison_records = []
    test_predictions = {}
    rf_importances = None

    for m_name, model in full_suite.items():
        print(f"[*] Training {m_name}...")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0
        
        preds = model.predict(X_te)
        test_predictions[m_name] = preds
        metrics = calculate_metrics(y_te, preds)
        
        print(f"    -> RMSE: {metrics['RMSE']:.4f} kW | MAE: {metrics['MAE']:.4f} kW | MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.4f} ({train_time:.2f}s)")
        
        comparison_records.append({
            'Model': m_name,
            'Algorithm_Type': 'From Scratch' if 'Scratch' in m_name else 'Benchmark GBDT',
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2'],
            'Train_Time_s': train_time
        })
        
        if m_name == 'Random Forest (Scratch)' and hasattr(model, 'feature_importances_'):
            rf_importances = model.feature_importances_

    comparison_df = pd.DataFrame(comparison_records)
    comp_csv_path = os.path.join(results_dir, "model_comparison_summary.csv")
    comparison_df.to_csv(comp_csv_path, index=False)
    print(f"\n[+] Saved Full Model Comparison to: {comp_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 6: CAPSTONE INVESTIGATION — FORECAST HORIZONS (E1 TO E5)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 6: CAPSTONE INVESTIGATION: FORECAST HORIZON EXPERIMENT (E1 - E5)")
    print("="*80)
    horizons = [
        ('E1 (10 Minutes)', 10),
        ('E2 (30 Minutes)', 30),
        ('E3 (1 Hour)', 60),
        ('E4 (6 Hours)', 360),
        ('E5 (24 Hours)', 1440)
    ]

    horizon_records = []
    
    # We evaluate degradation on Linear Regression, Random Forest, and GBDT across horizons
    for h_label, h_mins in horizons:
        print(f"\n---> Evaluating Horizon: {h_label} (+{h_mins} mins ahead)...")
        X_h, y_h, _ = build_experiment_features(
            df_clean, experiment_type='D', target_col='Global_active_power', horizon_minutes=h_mins
        )
        
        n_h = len(X_h)
        tr_end_h = int(0.70 * n_h)
        te_start_h = int(0.85 * n_h)
        
        X_tr_h = X_h.iloc[tr_end_h - train_sample_size:tr_end_h].values
        y_tr_h = y_h.iloc[tr_end_h - train_sample_size:tr_end_h].values
        
        X_te_h = X_h.iloc[te_start_h:te_start_h + test_sample_size].values
        y_te_h = y_h.iloc[te_start_h:te_start_h + test_sample_size].values
        
        suite_h = {
            'Linear Regression (Scratch)': LinearRegressionScratch(),
            'Decision Tree (Scratch)': DecisionTreeRegressorScratch(max_depth=10, min_samples_split=30),
            'Random Forest (Scratch)': RandomForestRegressorScratch(n_estimators=25, max_depth=10, min_samples_split=30),
            'Gradient Boosting (GBDT)': HistGradientBoostingRegressor(max_iter=60, max_depth=6, random_state=42)
        }
        
        for m_name, model in suite_h.items():
            model.fit(X_tr_h, y_tr_h)
            preds_h = model.predict(X_te_h)
            m_h = calculate_metrics(y_te_h, preds_h)
            print(f"     {m_name:28s} | RMSE: {m_h['RMSE']:.4f} kW | MAE: {m_h['MAE']:.4f} kW")
            
            horizon_records.append({
                'Horizon_Label': h_label,
                'Horizon_Minutes': h_mins,
                'Model': m_name,
                'RMSE': m_h['RMSE'],
                'MAE': m_h['MAE'],
                'MAPE': m_h['MAPE'],
                'R2': m_h['R2']
            })

    horizon_df = pd.DataFrame(horizon_records)
    horizon_csv_path = os.path.join(results_dir, "forecast_horizon_results.csv")
    horizon_df.to_csv(horizon_csv_path, index=False)
    print(f"\n[+] Saved Forecast Horizon Results to: {horizon_csv_path}")

    # -------------------------------------------------------------------------
    # STEP 7: GENERATE REMAINING EXPECTED GRAPHS (10 TO 14)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 7: GENERATING EXPECTED GRAPHS (10 TO 14)")
    print("="*80)

    # Graph 10: Actual vs Predicted
    plot_actual_vs_predicted(
        y_te, test_predictions, timestamps=test_timestamps,
        save_path=os.path.join(figures_dir, "fig10_actual_vs_predicted.png")
    )

    # Graph 11: Residual Distribution
    plot_residual_distribution(
        y_te, test_predictions,
        save_path=os.path.join(figures_dir, "fig11_residual_distribution.png")
    )

    # Graph 12: Forecast Horizon vs RMSE
    plot_forecast_horizon_vs_rmse(
        horizon_df,
        save_path=os.path.join(figures_dir, "fig12_forecast_horizon_vs_rmse.png")
    )

    # Graph 13: Feature Importance
    if rf_importances is not None:
        plot_feature_importance(
            full_feat_names, rf_importances, model_name="Random Forest (Scratch)",
            save_path=os.path.join(figures_dir, "fig13_feature_importance.png")
        )

    # Graph 14: Model Comparison
    plot_model_comparison(
        comparison_df,
        save_path=os.path.join(figures_dir, "fig14_model_comparison.png")
    )

    total_time = time.time() - start_total
    print("\n" + "="*80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f} SECONDS ({total_time/60:.2f} MINS)")
    print(f"All 14 Expected Figures Saved in: {figures_dir}/")
    print(f"All Result Tables Saved in:       {results_dir}/")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Household Electricity Consumption Prediction Pipeline")
    parser.add_argument("--dataset", type=str, default="dataset.txt", help="Path to raw dataset")
    parser.add_argument("--figures_dir", type=str, default="figures", help="Directory for plots")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for metric CSVs")
    parser.add_argument("--train_samples", type=int, default=60000, help="Train samples count")
    parser.add_argument("--test_samples", type=int, default=20000, help="Test samples count")
    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
        train_sample_size=args.train_samples,
        test_sample_size=args.test_samples
    )
