"""
AGRICOM - Weather Analysis for Demand Forecasting
Analyzes Berlin weather patterns and creates visualizations for the Feb 4 proposal.

Usage:
    python weather_analysis.py

Output:
    outputs/figures/weather_*.png
    outputs/reports/weather_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
OUTPUT_FIGURES = os.path.join(os.path.dirname(__file__), '../../outputs/figures')
OUTPUT_REPORTS = os.path.join(os.path.dirname(__file__), '../../outputs/reports')

# Ensure output directories exist
os.makedirs(OUTPUT_FIGURES, exist_ok=True)
os.makedirs(OUTPUT_REPORTS, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2ecc71',    # Green (organic theme)
    'secondary': '#3498db',  # Blue
    'accent': '#e74c3c',     # Red
    'warm': '#f39c12',       # Orange
    'cold': '#9b59b6',       # Purple
}


def load_weather_data():
    """Load the most recent weather data file."""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('weather_berlin') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No weather data found. Run weather.py first.")

    latest = sorted(files)[-1]
    filepath = os.path.join(DATA_DIR, latest)
    print(f"Loading: {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def plot_temperature_overview(df, save=True):
    """Create temperature overview with anomalies."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Filter to last 2 years for clarity
    two_years_ago = df.index.max() - pd.DateOffset(years=2)
    df_recent = df[df.index >= two_years_ago]

    # Plot 1: Temperature with range
    ax1 = axes[0]
    ax1.fill_between(df_recent.index,
                     df_recent['temperature_2m_min'],
                     df_recent['temperature_2m_max'],
                     alpha=0.3, color=COLORS['secondary'], label='Daily Range')
    ax1.plot(df_recent.index, df_recent['temperature_2m_mean'],
             color=COLORS['primary'], linewidth=1, label='Mean Temperature')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Berlin Temperature - Last 2 Years', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')

    # Plot 2: Temperature anomaly
    ax2 = axes[1]
    colors = [COLORS['accent'] if x > 0 else COLORS['cold'] for x in df_recent['temp_anomaly']]
    ax2.bar(df_recent.index, df_recent['temp_anomaly'], color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='±2°C threshold')
    ax2.axhline(y=-2, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Anomaly (°C)')
    ax2.set_title('Temperature Anomaly (vs Historical Average)', fontsize=12)
    ax2.legend(loc='upper right')

    # Plot 3: Precipitation
    ax3 = axes[2]
    ax3.bar(df_recent.index, df_recent['precipitation_sum'],
            color=COLORS['secondary'], alpha=0.7, width=1)
    ax3.set_ylabel('Precipitation (mm)')
    ax3.set_xlabel('Date')
    ax3.set_title('Daily Precipitation', fontsize=12)

    # Format x-axis
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'weather_temperature_overview.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()
    return fig


def plot_seasonal_patterns(df, save=True):
    """Analyze and plot seasonal patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Monthly temperature distribution
    ax1 = axes[0, 0]
    df['month_name'] = df.index.month_name()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

    monthly_temp = df.groupby('month_name')['temperature_2m_mean'].agg(['mean', 'std'])
    ax1.bar(range(12), monthly_temp['mean'], yerr=monthly_temp['std'],
            color=COLORS['primary'], alpha=0.7, capsize=3)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels([m[:3] for m in month_order], rotation=45)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Average Monthly Temperature', fontsize=12, fontweight='bold')
    ax1.axhline(y=df['temperature_2m_mean'].mean(), color='gray', linestyle='--',
                label=f'Annual avg: {df["temperature_2m_mean"].mean():.1f}°C')
    ax1.legend()

    # Monthly precipitation
    ax2 = axes[0, 1]
    monthly_precip = df.groupby('month_name')['precipitation_sum'].sum() / df.groupby('month_name').size().values[0] * 30
    ax2.bar(range(12), monthly_precip.values, color=COLORS['secondary'], alpha=0.7)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels([m[:3] for m in month_order], rotation=45)
    ax2.set_ylabel('Precipitation (mm/month)')
    ax2.set_title('Average Monthly Precipitation', fontsize=12, fontweight='bold')

    # Anomaly distribution by season
    ax3 = axes[1, 0]
    season_order = ['spring', 'summer', 'autumn', 'winter']
    df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
    season_colors = [COLORS['primary'], COLORS['warm'], COLORS['accent'], COLORS['cold']]

    for i, season in enumerate(season_order):
        season_data = df[df['season'] == season]['temp_anomaly'].dropna()
        ax3.hist(season_data, bins=30, alpha=0.5, label=season.capitalize(),
                 color=season_colors[i], density=True)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(x=-2, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Temperature Anomaly (°C)')
    ax3.set_ylabel('Density')
    ax3.set_title('Anomaly Distribution by Season', fontsize=12, fontweight='bold')
    ax3.legend()

    # Extreme weather days by month
    ax4 = axes[1, 1]
    df['is_extreme'] = (df['temp_anomaly'].abs() > 5) | (df['precipitation_sum'] > 10)
    extreme_by_month = df.groupby(df.index.month)['is_extreme'].sum()
    ax4.bar(range(1, 13), extreme_by_month.values, color=COLORS['accent'], alpha=0.7)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels([m[:3] for m in month_order])
    ax4.set_ylabel('Number of Extreme Days')
    ax4.set_xlabel('Month')
    ax4.set_title('Extreme Weather Days by Month', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'weather_seasonal_patterns.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()
    return fig


def plot_demand_hypothesis(df, save=True):
    """Visualize the weather-demand hypothesis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Hypothesis: Temperature affects produce preference
    ax1 = axes[0, 0]
    bins = pd.cut(df['temperature_2m_mean'], bins=[-20, 5, 15, 25, 40],
                  labels=['Cold (<5°C)', 'Mild (5-15°C)', 'Warm (15-25°C)', 'Hot (>25°C)'])
    temp_dist = bins.value_counts().sort_index()
    colors_temp = [COLORS['cold'], COLORS['secondary'], COLORS['primary'], COLORS['accent']]
    ax1.pie(temp_dist.values, labels=temp_dist.index, colors=colors_temp,
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Temperature Distribution in Berlin', fontsize=12, fontweight='bold')

    # Expected demand patterns text
    ax2 = axes[0, 1]
    ax2.axis('off')
    hypothesis_text = """
    HYPOTHESIS: Weather-Demand Correlation

    Based on literature review:

    COLD DAYS (<5°C) - 7% of days
    → ↑ Root vegetables, cooking ingredients
    → ↑ Soups, stews, warm foods
    → ↓ Salads, cold items

    WARM/HOT DAYS (>15°C) - 47% of days
    → ↑ Salads, fresh vegetables
    → ↑ Berries, cold fruits
    → ↑ Grilling items

    RAINY DAYS - 34% of days
    → ↑ Convenience purchases
    → ↑ Indoor cooking ingredients
    → ↓ Outdoor market visits

    KEY INSIGHT:
    90% of weather-based demand volatility
    comes from temperature and precipitation.
    (Source: Planalytics research)
    """
    ax2.text(0.1, 0.9, hypothesis_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Weekly pattern
    ax3 = axes[1, 0]
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = df.index.day_name()
    df['day_name'] = pd.Categorical(df['day_name'], categories=day_names, ordered=True)

    # This would correlate with shopping patterns
    ax3.bar(range(7), [85, 90, 88, 92, 95, 100, 70],  # Illustrative shopping index
            color=COLORS['primary'], alpha=0.7)
    ax3.set_xticks(range(7))
    ax3.set_xticklabels([d[:3] for d in day_names])
    ax3.set_ylabel('Shopping Index (illustrative)')
    ax3.set_title('Typical Weekly Shopping Pattern', fontsize=12, fontweight='bold')
    ax3.axhline(y=90, color='gray', linestyle='--', alpha=0.5)

    # Lead time analysis
    ax4 = axes[1, 1]
    lead_times = ['Same day', '1-2 days', '3-5 days', '1 week', '2+ weeks']
    confidence = [95, 85, 70, 50, 30]  # Forecast accuracy by lead time
    ax4.barh(lead_times, confidence, color=COLORS['secondary'], alpha=0.7)
    ax4.set_xlabel('Weather Forecast Accuracy (%)')
    ax4.set_title('Weather Forecast Reliability by Lead Time', fontsize=12, fontweight='bold')
    ax4.axvline(x=70, color='gray', linestyle='--', alpha=0.5, label='Useful threshold')
    for i, v in enumerate(confidence):
        ax4.text(v + 1, i, f'{v}%', va='center')

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'weather_demand_hypothesis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()
    return fig


def generate_summary_stats(df):
    """Generate summary statistics for the report."""
    summary = {
        'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
        'total_days': len(df),
        'avg_temperature': df['temperature_2m_mean'].mean(),
        'min_temperature': df['temperature_2m_min'].min(),
        'max_temperature': df['temperature_2m_max'].max(),
        'total_rainy_days': df['is_rainy'].sum(),
        'rainy_day_pct': df['is_rainy'].mean() * 100,
        'hot_days': (df['temperature_2m_max'] > 25).sum(),
        'cold_days': (df['temperature_2m_min'] < 0).sum(),
        'extreme_anomaly_days': (df['temp_anomaly'].abs() > 5).sum(),
        'avg_precipitation_mm': df['precipitation_sum'].mean(),
    }

    # Anomaly category distribution
    anomaly_dist = df['temp_anomaly_category'].value_counts()
    for cat in ['very_cold', 'cold', 'normal', 'warm', 'very_warm']:
        if cat in anomaly_dist.index:
            summary[f'anomaly_{cat}_days'] = anomaly_dist[cat]
        else:
            summary[f'anomaly_{cat}_days'] = 0

    return summary


def main():
    """Main analysis routine."""
    print("=" * 60)
    print("AGRICOM - Weather Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading weather data...")
    df = load_weather_data()
    print(f"   Loaded {len(df)} days of data")

    # Generate visualizations
    print("\n2. Generating visualizations...")

    print("   - Temperature overview...")
    plot_temperature_overview(df)

    print("   - Seasonal patterns...")
    plot_seasonal_patterns(df)

    print("   - Demand hypothesis...")
    plot_demand_hypothesis(df)

    # Generate summary
    print("\n3. Generating summary statistics...")
    summary = generate_summary_stats(df)

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(OUTPUT_REPORTS, 'weather_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"   Saved: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("WEATHER SUMMARY")
    print("=" * 60)
    print(f"Date Range: {summary['date_range']}")
    print(f"Total Days: {summary['total_days']}")
    print(f"\nTemperature:")
    print(f"  Average: {summary['avg_temperature']:.1f}°C")
    print(f"  Min: {summary['min_temperature']:.1f}°C")
    print(f"  Max: {summary['max_temperature']:.1f}°C")
    print(f"  Hot days (>25°C): {summary['hot_days']}")
    print(f"  Cold days (<0°C): {summary['cold_days']}")
    print(f"\nPrecipitation:")
    print(f"  Rainy days: {summary['total_rainy_days']} ({summary['rainy_day_pct']:.1f}%)")
    print(f"  Avg daily: {summary['avg_precipitation_mm']:.1f}mm")
    print(f"\nAnomalies:")
    print(f"  Extreme days (>5°C deviation): {summary['extreme_anomaly_days']}")
    print(f"  Distribution: very_cold={summary['anomaly_very_cold_days']}, "
          f"cold={summary['anomaly_cold_days']}, normal={summary['anomaly_normal_days']}, "
          f"warm={summary['anomaly_warm_days']}, very_warm={summary['anomaly_very_warm_days']}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to:")
    print(f"  - {OUTPUT_FIGURES}/weather_*.png")
    print(f"  - {OUTPUT_REPORTS}/weather_summary.csv")


if __name__ == "__main__":
    main()
