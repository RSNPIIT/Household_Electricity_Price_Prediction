"""
Feature Engineering Module for Household Electricity Consumption Prediction.
Constructs temporal, cyclical, lag, and rolling statistical features with strict
time-series causality (zero future data leakage).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def create_temporal_features(df: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
    """
    Extract calendar and diurnal temporal features from the datetime index.
    
    Features created:
    - Hour (0-23)
    - Minute (0-59)
    - Day (1-31)
    - Day_of_week (0-6)
    - Month (1-12)
    - Is_weekend (0 or 1)
    - Cyclical sin/cos encodings for hour, minute, day of week, and month.
    """
    df_feat = df.copy()
    dt = df_feat.index
    
    df_feat['Hour'] = dt.hour
    df_feat['Minute'] = dt.minute
    df_feat['Day'] = dt.day
    df_feat['Day_of_week'] = dt.dayofweek
    df_feat['Month'] = dt.month
    df_feat['Is_weekend'] = (df_feat['Day_of_week'] >= 5).astype(int)
    
    if cyclical:
        # 24-hour cycle
        df_feat['Hour_sin'] = np.sin(2 * np.pi * df_feat['Hour'] / 24.0)
        df_feat['Hour_cos'] = np.cos(2 * np.pi * df_feat['Hour'] / 24.0)
        # 60-minute cycle
        df_feat['Minute_sin'] = np.sin(2 * np.pi * df_feat['Minute'] / 60.0)
        df_feat['Minute_cos'] = np.cos(2 * np.pi * df_feat['Minute'] / 60.0)
        # 7-day weekly cycle
        df_feat['DayOfWeek_sin'] = np.sin(2 * np.pi * df_feat['Day_of_week'] / 7.0)
        df_feat['DayOfWeek_cos'] = np.cos(2 * np.pi * df_feat['Day_of_week'] / 7.0)
        # 12-month annual cycle
        df_feat['Month_sin'] = np.sin(2 * np.pi * df_feat['Month'] / 12.0)
        df_feat['Month_cos'] = np.cos(2 * np.pi * df_feat['Month'] / 12.0)
        
    return df_feat


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    lags: List[int] = [1, 10, 30, 60, 1440]
) -> pd.DataFrame:
    """
    Engineer autoregressive lag features for the target variable and key electrical covariates.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with datetime index.
    target_col : str
        Primary target variable to lag.
    lags : list of int
        Lag steps in minutes (e.g. 1=1m, 10=10m, 30=30m, 60=1h, 1440=24h).
    """
    df_feat = df.copy()
    
    for lag in lags:
        # Target lag
        df_feat[f'{target_col}_lag_{lag}m'] = df_feat[target_col].shift(lag)
        
    # Also add key electrical lags for Voltage and Intensity (1m, 10m, 60m)
    for col in ['Global_intensity', 'Voltage']:
        if col in df_feat.columns:
            for lag in [1, 10, 60]:
                df_feat[f'{col}_lag_{lag}m'] = df_feat[col].shift(lag)
                
    return df_feat


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'Global_active_power',
    windows: List[int] = [10, 30, 60]
) -> pd.DataFrame:
    """
    Engineer rolling statistical features (mean, std, min, max) over historical windows.
    
    LEAKAGE-FREE GUARANTEE:
    Shift the target by 1 minute before computing rolling statistics so that the
    current observation y_t is strictly NOT included in the window!
    """
    df_feat = df.copy()
    shifted_target = df_feat[target_col].shift(1)
    
    for w in windows:
        roll = shifted_target.rolling(window=w, min_periods=max(1, w // 2))
        df_feat[f'{target_col}_roll_mean_{w}m'] = roll.mean()
        df_feat[f'{target_col}_roll_std_{w}m'] = roll.std().fillna(0.0)
        df_feat[f'{target_col}_roll_min_{w}m'] = roll.min()
        df_feat[f'{target_col}_roll_max_{w}m'] = roll.max()
        
    return df_feat


def build_experiment_features(
    df: pd.DataFrame,
    experiment_type: str = 'D',
    target_col: str = 'Global_active_power',
    horizon_minutes: int = 1
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build feature matrices for Experiment A, B, C, D, or E.
    
    Experiment A: Baseline (available electrical variables only: Reactive Power, Voltage, Intensity, Sub-meterings)
    Experiment B: Temporal (+ Hour, Minute, Day, Day_of_week, Month, Weekend, Cyclical)
    Experiment C: Lag Features (+ Lags 1m, 10m, 30m, 1h, 24h)
    Experiment D: Rolling Statistics (+ Rolling Mean, Std, Min, Max over 10m, 30m, 60m)
    Experiment E: Forecast Horizon comparison using the full feature set (D)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned dataframe.
    experiment_type : str
        'A', 'B', 'C', 'D', or 'E'
    target_col : str
        Target variable name ('Global_active_power')
    horizon_minutes : int
        Prediction horizon in minutes (e.g. 1, 10, 30, 60, 360, 1440).
        The target becomes y_{t + horizon_minutes}.
    """
    df_work = df.copy()
    
    # Define future target: y(t + horizon_minutes)
    if horizon_minutes == 1:
        # Standard 1-step-ahead prediction: predict next minute
        target_series = df_work[target_col].shift(-1)
    else:
        # Predict t + horizon_minutes
        target_series = df_work[target_col].shift(-horizon_minutes)
        
    df_work['Target_Horizon'] = target_series

    # Base electrical variables (known at time t)
    base_electrical = [
        'Global_reactive_power', 'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'Sub_metering_remainder'
    ]
    
    if experiment_type == 'A':
        feature_cols = [c for c in base_electrical if c in df_work.columns]
        
    elif experiment_type == 'B':
        df_work = create_temporal_features(df_work, cyclical=True)
        temporal_cols = [
            'Hour', 'Minute', 'Day', 'Day_of_week', 'Month', 'Is_weekend',
            'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
            'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos'
        ]
        feature_cols = [c for c in base_electrical if c in df_work.columns] + temporal_cols
        
    elif experiment_type == 'C':
        df_work = create_temporal_features(df_work, cyclical=True)
        df_work = create_lag_features(df_work, target_col=target_col, lags=[1, 10, 30, 60, 1440])
        lag_cols = [c for c in df_work.columns if '_lag_' in c]
        temporal_cols = [
            'Hour', 'Minute', 'Day', 'Day_of_week', 'Month', 'Is_weekend',
            'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
            'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos'
        ]
        feature_cols = [c for c in base_electrical if c in df_work.columns] + temporal_cols + lag_cols
        
    elif experiment_type in ['D', 'E']:
        df_work = create_temporal_features(df_work, cyclical=True)
        df_work = create_lag_features(df_work, target_col=target_col, lags=[1, 10, 30, 60, 1440])
        df_work = create_rolling_features(df_work, target_col=target_col, windows=[10, 30, 60])
        
        lag_cols = [c for c in df_work.columns if '_lag_' in c]
        roll_cols = [c for c in df_work.columns if '_roll_' in c]
        temporal_cols = [
            'Hour', 'Minute', 'Day', 'Day_of_week', 'Month', 'Is_weekend',
            'Hour_sin', 'Hour_cos', 'Minute_sin', 'Minute_cos',
            'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos'
        ]
        feature_cols = [c for c in base_electrical if c in df_work.columns] + temporal_cols + lag_cols + roll_cols
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    # Drop rows with NaNs caused by lags (max 1440) and future horizon
    df_valid = df_work.dropna(subset=feature_cols + ['Target_Horizon']).copy()
    
    X = df_valid[feature_cols]
    y = df_valid['Target_Horizon']
    
    return X, y, feature_cols
