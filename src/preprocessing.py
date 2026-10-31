"""
Data Preprocessing & Cleaning Module for Household Electricity Consumption Prediction.
Handles ingestion of 2,075,259 observations, datetime parsing, missing value imputation,
and chronological train/validation/test splitting.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional


def load_raw_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the raw UCI Household Electric Power Consumption dataset.
    
    Parameters
    ----------
    filepath : str
        Path to dataset.txt (semicolon-separated).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed datetime index and float typed measurements.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    print(f"[*] Ingesting raw dataset from: {filepath}...")
    df = pd.read_csv(
        filepath,
        sep=';',
        na_values=['?'],
        dtype={
            'Global_active_power': 'float64',
            'Global_reactive_power': 'float64',
            'Voltage': 'float64',
            'Global_intensity': 'float64',
            'Sub_metering_1': 'float64',
            'Sub_metering_2': 'float64',
            'Sub_metering_3': 'float64'
        },
        low_memory=False
    )
    
    # Construct unified datetime index
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"[*] Successfully loaded {len(df):,} rows spanning {df.index.min()} to {df.index.max()}.")
    return df


def audit_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit missing observations across all electrical variables.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100.0
    audit_df = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_pct
    })
    return audit_df


def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing observations in the time series using leakage-free imputation.
    
    For time-series data, forward-fill ('ffill') replicates the last known sensor
    reading into subsequent missing timestamps, strictly preserving causal ordering.
    """
    df_clean = df.copy()
    initial_missing = df_clean.isnull().sum().sum()
    
    if method == 'ffill':
        # Forward fill last known sensor reading, followed by backward fill for any leading NaNs
        df_clean = df_clean.ffill().bfill()
    elif method == 'interpolate':
        # Time-based linear interpolation
        df_clean = df_clean.interpolate(method='time').bfill()
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    remaining_missing = df_clean.isnull().sum().sum()
    print(f"[*] Imputed {initial_missing:,} missing values via '{method}'. Remaining NaNs: {remaining_missing}")
    
    # Calculate remainder sub-metering (unmetered active energy in Watt-hours)
    # Energy per minute = (Global_active_power * 1000 / 60) - (Sub_metering_1 + Sub_metering_2 + Sub_metering_3)
    active_energy_wh = (df_clean['Global_active_power'] * 1000.0 / 60.0)
    sub_metering_sum = df_clean['Sub_metering_1'] + df_clean['Sub_metering_2'] + df_clean['Sub_metering_3']
    df_clean['Sub_metering_remainder'] = np.maximum(0.0, active_energy_wh - sub_metering_sum)
    
    return df_clean


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/validation/test split without random shuffling.
    
    Strictly preserves temporal causality:
    - Train: [0 to T_train]
    - Validation: [T_train to T_val]
    - Test: [T_val to End]
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"[*] Chronological Split Summary:")
    print(f"    Train Set:      {len(df_train):,} samples ({df_train.index.min()} to {df_train.index.max()})")
    print(f"    Validation Set: {len(df_val):,} samples ({df_val.index.min()} to {df_val.index.max()})")
    print(f"    Test Set:       {len(df_test):,} samples ({df_test.index.min()} to {df_test.index.max()})")
    
    return df_train, df_val, df_test
