"""
Exploratory Data Analysis (EDA) & Visualization Module.
Generates all 9 specified EDA figures:
1. Global active power distribution
2. Power consumption over time
3. Hourly consumption pattern
4. Daily consumption pattern
5. Weekly consumption pattern
6. Monthly consumption pattern
7. Correlation heatmap
8. Lag correlation plot (ACF & PACF)
9. Rolling mean plot
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional


# Set global clean academic plotting style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300


def plot_active_power_distribution(df: pd.DataFrame, save_path: str = "figures/fig01_active_power_distribution.png"):
    """
    Graph 1: Global active power distribution (Histogram + KDE).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    data = df['Global_active_power'].dropna()
    sns.histplot(data, bins=80, kde=True, color='#2563eb', stat="density", alpha=0.6, edgecolor='white')
    
    mean_val = data.mean()
    median_val = data.median()
    p95_val = data.quantile(0.95)
    
    plt.axvline(mean_val, color='#dc2626', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f} kW')
    plt.axvline(median_val, color='#059669', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f} kW')
    plt.axvline(p95_val, color='#d97706', linestyle=':', linewidth=1.5, label=f'95th Percentile: {p95_val:.2f} kW')
    
    plt.title("Distribution of Global Active Power Consumption (2006–2010)", pad=14, weight='bold')
    plt.xlabel("Global Active Power (kW)")
    plt.ylabel("Density")
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 1: {save_path}")


def plot_power_over_time(df: pd.DataFrame, save_path: str = "figures/fig02_power_consumption_time_series.png"):
    """
    Graph 2: Power consumption over time (Daily and 30-Day moving average trends).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 5))
    
    # Resample to daily mean for clear temporal visualization
    daily_power = df['Global_active_power'].resample('D').mean()
    rolling_30d = daily_power.rolling(window=30, center=True).mean()
    
    plt.plot(daily_power.index, daily_power.values, color='#93c5fd', alpha=0.5, linewidth=1.0, label='Daily Average Power')
    plt.plot(rolling_30d.index, rolling_30d.values, color='#1e3a8a', linewidth=2.0, label='30-Day Moving Average (Trend)')
    
    plt.title("Household Electricity Consumption Over Time (4-Year Horizon)", pad=14, weight='bold')
    plt.xlabel("Timeline")
    plt.ylabel("Global Active Power (kW)")
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 2: {save_path}")


def plot_hourly_consumption_pattern(df: pd.DataFrame, save_path: str = "figures/fig03_hourly_consumption_pattern.png"):
    """
    Graph 3: Hourly consumption pattern (Diurnal cycle with morning and evening peaks).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    df_temp = df.copy()
    df_temp['Hour'] = df_temp.index.hour
    hourly_stats = df_temp.groupby('Hour')['Global_active_power'].agg(['mean', 'std', 'median'])
    
    hours = hourly_stats.index
    means = hourly_stats['mean']
    stds = hourly_stats['std']
    
    plt.plot(hours, means, marker='o', color='#0284c7', linewidth=2.2, label='Mean Consumption')
    plt.fill_between(hours, np.maximum(0, means - stds), means + stds, color='#0284c7', alpha=0.2, label='±1 Standard Deviation')
    plt.plot(hours, hourly_stats['median'], marker='s', linestyle='--', color='#059669', linewidth=1.8, label='Median Consumption')
    
    plt.title("Diurnal Electricity Consumption Pattern (24-Hour Cycle)", pad=14, weight='bold')
    plt.xlabel("Hour of the Day (0–23)")
    plt.ylabel("Global Active Power (kW)")
    plt.xticks(range(0, 24))
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 3: {save_path}")


def plot_daily_consumption_pattern(df: pd.DataFrame, save_path: str = "figures/fig04_daily_consumption_pattern.png"):
    """
    Graph 4: Daily consumption pattern (Day of month variations).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(11, 5))
    
    df_temp = df.copy()
    df_temp['Day'] = df_temp.index.day
    daily_stats = df_temp.groupby('Day')['Global_active_power'].agg(['mean', 'std'])
    
    days = daily_stats.index
    means = daily_stats['mean']
    
    plt.bar(days, means, color='#6366f1', alpha=0.75, edgecolor='#4338ca', width=0.65, label='Mean Daily Power')
    plt.axhline(means.mean(), color='#dc2626', linestyle='--', linewidth=1.5, label=f'Grand Mean: {means.mean():.2f} kW')
    
    plt.title("Average Electricity Consumption by Day of Month (Days 1–31)", pad=14, weight='bold')
    plt.xlabel("Day of the Month")
    plt.ylabel("Global Active Power (kW)")
    plt.xticks(range(1, 32))
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 4: {save_path}")


def plot_weekly_consumption_pattern(df: pd.DataFrame, save_path: str = "figures/fig05_weekly_consumption_pattern.png"):
    """
    Graph 5: Weekly consumption pattern (Weekday vs. Weekend).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    # Resample only active power to hourly mean
    hourly = df['Global_active_power'].resample('1h').mean().to_frame()
    hourly['DayOfWeek'] = hourly.index.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    palette = ['#0284c7']*5 + ['#ea580c', '#ea580c'] # highlight weekend in warm orange
    sns.boxplot(data=hourly, x='DayOfWeek', y='Global_active_power',
                order=order, palette=palette, showfliers=False)
    
    plt.title("Weekly Power Consumption: Weekdays vs. Weekends (Hourly Resampled)", pad=14, weight='bold')
    plt.xlabel("Day of the Week")
    plt.ylabel("Hourly Active Power (kW)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 5: {save_path}")


def plot_monthly_consumption_pattern(df: pd.DataFrame, save_path: str = "figures/fig06_monthly_consumption_pattern.png"):
    """
    Graph 6: Monthly consumption pattern (Seasonal heating/cooling dynamics).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(11, 5))
    
    df_temp = df.copy()
    df_temp['Month'] = df_temp.index.month
    monthly_stats = df_temp.groupby('Month')['Global_active_power'].agg(['mean', 'std'])
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months = np.arange(1, 13)
    means = monthly_stats['mean']
    
    # Warm winter colors vs cool summer colors
    colors = ['#dc2626', '#dc2626', '#ea580c', '#f59e0b', '#10b981', '#06b6d4',
              '#0284c7', '#0284c7', '#10b981', '#f59e0b', '#ea580c', '#dc2626']
    
    plt.bar(months, means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.title("Seasonal Electricity Consumption Profile Across Calendar Months", pad=14, weight='bold')
    plt.xlabel("Month")
    plt.ylabel("Average Active Power (kW)")
    plt.xticks(months, month_names)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 6: {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = "figures/fig07_correlation_heatmap.png"):
    """
    Graph 7: Correlation heatmap of all electrical measurements.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 7))
    
    cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_remainder'
    ]
    valid_cols = [c for c in cols if c in df.columns]
    corr = df[valid_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt=".2f",
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title("Correlation Matrix of Electrical Features & Sub-Meterings", pad=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 7: {save_path}")


def plot_lag_correlation(df: pd.DataFrame, max_lag: int = 60, save_path: str = "figures/fig08_lag_correlation_acf_pacf.png"):
    """
    Graph 8: Autocorrelation (ACF) plot across lags (1 to 60 minutes).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    
    series = df['Global_active_power'].dropna()
    # Compute autocorrelations for lags 1 through max_lag
    autocorrs = [series.autocorr(lag=l) for l in range(1, max_lag + 1)]
    lags = range(1, max_lag + 1)
    
    ax.bar(lags, autocorrs, color='#0284c7', alpha=0.7, width=0.6)
    ax.plot(lags, autocorrs, color='#0369a1', linewidth=1.5)
    
    # 95% confidence bounds: +- 1.96 / sqrt(N)
    ci = 1.96 / np.sqrt(len(series))
    ax.axhline(ci, color='#dc2626', linestyle='--', linewidth=1.0, label='95% Significance Limit')
    ax.axhline(-ci, color='#dc2626', linestyle='--', linewidth=1.0)
    ax.axhline(0, color='gray', linewidth=0.8)
    
    ax.set_title(f"Autocorrelation of Global Active Power Across {max_lag} Minutes", pad=14, weight='bold')
    ax.set_xlabel("Lag (Minutes)")
    ax.set_ylabel("Autocorrelation Coefficient")
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 8: {save_path}")


def plot_rolling_mean_comparison(df: pd.DataFrame, save_path: str = "figures/fig09_rolling_mean_plot.png"):
    """
    Graph 9: Rolling mean plot comparing raw 1-minute signal vs 1-hour and 24-hour averages.
    Plotted over a clean 2-week window for granular visibility.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 5))
    
    # Representative 2-week window in winter
    sample_df = df.loc['2007-01-01':'2007-01-14', 'Global_active_power']
    roll_1h = sample_df.rolling(window=60, center=True).mean()
    roll_24h = sample_df.rolling(window=1440, center=True).mean()
    
    plt.plot(sample_df.index, sample_df.values, color='#cbd5e1', alpha=0.8, linewidth=0.8, label='Raw 1-Minute Active Power')
    plt.plot(roll_1h.index, roll_1h.values, color='#0284c7', linewidth=1.8, label='1-Hour Rolling Mean (60m)')
    plt.plot(roll_24h.index, roll_24h.values, color='#dc2626', linewidth=2.5, label='24-Hour Rolling Mean (1440m)')
    
    plt.title("Signal Smoothing Comparison: Raw 1-Min vs. 1-Hour vs. 24-Hour Rolling Averages", pad=14, weight='bold')
    plt.xlabel("Timeline (Jan 2007 Sample)")
    plt.ylabel("Global Active Power (kW)")
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Graph 9: {save_path}")


def generate_all_eda_plots(df: pd.DataFrame, figures_dir: str = "figures"):
    """
    Generate and save all 9 EDA plots in sequence.
    """
    print("\n" + "="*70)
    print("EXECUTING TEMPORAL EDA & GENERATING EXPECTED GRAPHS (1 TO 9)")
    print("="*70)
    plot_active_power_distribution(df, os.path.join(figures_dir, "fig01_active_power_distribution.png"))
    plot_power_over_time(df, os.path.join(figures_dir, "fig02_power_consumption_time_series.png"))
    plot_hourly_consumption_pattern(df, os.path.join(figures_dir, "fig03_hourly_consumption_pattern.png"))
    plot_daily_consumption_pattern(df, os.path.join(figures_dir, "fig04_daily_consumption_pattern.png"))
    plot_weekly_consumption_pattern(df, os.path.join(figures_dir, "fig05_weekly_consumption_pattern.png"))
    plot_monthly_consumption_pattern(df, os.path.join(figures_dir, "fig06_monthly_consumption_pattern.png"))
    plot_correlation_heatmap(df, os.path.join(figures_dir, "fig07_correlation_heatmap.png"))
    plot_lag_correlation(df, max_lag=60, save_path=os.path.join(figures_dir, "fig08_lag_correlation_acf_pacf.png"))
    plot_rolling_mean_comparison(df, os.path.join(figures_dir, "fig09_rolling_mean_plot.png"))
    print("[*] All 9 EDA graphs generated successfully!\n")
