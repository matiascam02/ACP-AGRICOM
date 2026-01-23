"""
AGRICOM - Correlation Analysis
Analyzes relationships between external signals and demand proxies.

Usage:
    python correlation_analysis.py

Output:
    outputs/figures/correlation_*.png
    outputs/reports/correlation_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import correlate
import os
from datetime import datetime

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
OUTPUT_FIGURES = os.path.join(os.path.dirname(__file__), '../../outputs/figures')
OUTPUT_REPORTS = os.path.join(os.path.dirname(__file__), '../../outputs/reports')

# Ensure output directories exist
os.makedirs(OUTPUT_FIGURES, exist_ok=True)
os.makedirs(OUTPUT_REPORTS, exist_ok=True)


def load_latest_data(prefix):
    """Load the most recent data file with given prefix."""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix) and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No files found with prefix: {prefix}")

    latest = sorted(files)[-1]
    filepath = os.path.join(DATA_DIR, latest)
    print(f"Loading: {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def cross_correlation(series1, series2, max_lag=12):
    """
    Calculate cross-correlation between two time series at different lags.

    Args:
        series1: First time series (e.g., signal)
        series2: Second time series (e.g., demand proxy)
        max_lag: Maximum lag in weeks to test

    Returns:
        DataFrame with lag and correlation values
    """
    # Align series
    aligned = pd.concat([series1, series2], axis=1).dropna()
    s1 = aligned.iloc[:, 0].values
    s2 = aligned.iloc[:, 1].values

    # Standardize
    s1 = (s1 - s1.mean()) / s1.std()
    s2 = (s2 - s2.mean()) / s2.std()

    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
        else:
            corr = np.corrcoef(s1, s2)[0, 1]

        results.append({'lag_weeks': lag, 'correlation': corr})

    return pd.DataFrame(results)


def granger_causality_test(series1, series2, max_lag=4):
    """
    Test for Granger causality between two series.

    Args:
        series1: Potential cause series
        series2: Potential effect series
        max_lag: Maximum lag to test

    Returns:
        Dictionary with test results
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    # Align and prepare data
    aligned = pd.concat([series2, series1], axis=1).dropna()

    if len(aligned) < max_lag * 2 + 10:
        return {'error': 'Insufficient data for Granger test'}

    try:
        results = grangercausalitytests(aligned, maxlag=max_lag, verbose=False)

        # Extract p-values for each lag
        p_values = {}
        for lag in range(1, max_lag + 1):
            p_values[f'lag_{lag}'] = results[lag][0]['ssr_ftest'][1]

        return {
            'min_p_value': min(p_values.values()),
            'best_lag': min(p_values, key=p_values.get),
            'p_values': p_values,
            'significant': min(p_values.values()) < 0.05
        }
    except Exception as e:
        return {'error': str(e)}


def plot_cross_correlation(corr_df, title, save_path):
    """Create cross-correlation plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot bars
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in corr_df['correlation']]
    ax.bar(corr_df['lag_weeks'], corr_df['correlation'], color=colors, alpha=0.7)

    # Highlight max correlation
    max_idx = corr_df['correlation'].abs().idxmax()
    max_lag = corr_df.loc[max_idx, 'lag_weeks']
    max_corr = corr_df.loc[max_idx, 'correlation']

    ax.annotate(f'Max: {max_corr:.3f} at lag {max_lag}',
                xy=(max_lag, max_corr),
                xytext=(max_lag + 2, max_corr + 0.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)

    # Significance lines (approximate)
    n = len(corr_df)
    sig_level = 1.96 / np.sqrt(n)
    ax.axhline(y=sig_level, color='gray', linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(y=-sig_level, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xlabel('Lag (weeks)')
    ax.set_ylabel('Correlation')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_time_series_comparison(df, col1, col2, title, save_path):
    """Create dual-axis time series plot."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # First series
    color1 = '#3498db'
    ax1.plot(df.index, df[col1], color=color1, linewidth=1.5, label=col1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(col1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Second series (secondary axis)
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.plot(df.index, df[col2], color=color2, linewidth=1.5, label=col2, alpha=0.7)
    ax2.set_ylabel(col2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and legend
    fig.suptitle(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def analyze_weather_trends_correlation(weather_df, trends_df):
    """
    Analyze correlation between weather and Google Trends data.

    Args:
        weather_df: Weather data
        trends_df: Google Trends data

    Returns:
        Summary DataFrame
    """
    results = []

    # Resample weather to weekly (trends are weekly)
    weather_weekly = weather_df.resample('W').agg({
        'temperature_2m_mean': 'mean',
        'temp_anomaly': 'mean',
        'precipitation_sum': 'sum',
        'is_rainy': 'sum'
    })

    # Get trend columns (exclude metadata)
    trend_cols = [c for c in trends_df.columns if not c.startswith('is') and c != 'category']

    for trend_col in trend_cols:
        # Align data
        combined = pd.concat([
            weather_weekly[['temperature_2m_mean', 'temp_anomaly']],
            trends_df[[trend_col]]
        ], axis=1).dropna()

        if len(combined) < 20:
            continue

        # Calculate correlations
        for weather_var in ['temperature_2m_mean', 'temp_anomaly']:
            corr, p_value = stats.pearsonr(combined[weather_var], combined[trend_col])

            results.append({
                'trend_keyword': trend_col,
                'weather_variable': weather_var,
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_observations': len(combined)
            })

    return pd.DataFrame(results)


def main():
    """Main analysis routine."""
    print("=" * 60)
    print("AGRICOM - Correlation Analysis")
    print("=" * 60)

    try:
        # Load data
        print("\n1. Loading data...")
        weather_df = load_latest_data('weather_berlin')
        trends_df = load_latest_data('google_trends_berlin')

        print(f"   Weather: {len(weather_df)} days")
        print(f"   Trends: {len(trends_df)} weeks")

        # Basic correlations
        print("\n2. Calculating weather-trends correlations...")
        corr_results = analyze_weather_trends_correlation(weather_df, trends_df)

        if not corr_results.empty:
            # Save results
            report_path = os.path.join(OUTPUT_REPORTS, 'correlation_summary.csv')
            corr_results.to_csv(report_path, index=False)
            print(f"\n   Saved: {report_path}")

            # Print top correlations
            print("\n   Top correlations (by absolute value):")
            top = corr_results.nlargest(10, 'correlation')
            for _, row in top.iterrows():
                sig = '*' if row['significant'] else ''
                print(f"   {row['trend_keyword']} vs {row['weather_variable']}: "
                      f"{row['correlation']:.3f}{sig}")

        # Generate sample visualizations
        print("\n3. Generating visualizations...")

        # Weather time series
        if 'temperature_2m_mean' in weather_df.columns:
            fig_path = os.path.join(OUTPUT_FIGURES, 'weather_overview.png')

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            axes[0].plot(weather_df.index, weather_df['temperature_2m_mean'],
                        color='#e74c3c', linewidth=0.5)
            axes[0].fill_between(weather_df.index,
                                weather_df['temperature_2m_min'],
                                weather_df['temperature_2m_max'],
                                alpha=0.3, color='#e74c3c')
            axes[0].set_ylabel('Temperature (°C)')
            axes[0].set_title('Berlin Temperature')

            axes[1].bar(weather_df.index, weather_df['precipitation_sum'],
                       color='#3498db', alpha=0.7, width=1)
            axes[1].set_ylabel('Precipitation (mm)')
            axes[1].set_xlabel('Date')

            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"   Saved: {fig_path}")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run data collection scripts first:")
        print("  python src/data_collection/google_trends.py")
        print("  python src/data_collection/weather.py")


if __name__ == "__main__":
    main()
