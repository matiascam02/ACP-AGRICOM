"""
AGRICOM - Google Trends + Weather Correlation Analysis (Fixed)
Analyzes relationship between search behavior and weather patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
TRENDS_DIR = os.path.join(DATA_DIR, 'google_trends')
OUTPUT_FIGURES = os.path.join(os.path.dirname(__file__), '../../outputs/figures')
OUTPUT_REPORTS = os.path.join(os.path.dirname(__file__), '../../outputs/reports')

os.makedirs(OUTPUT_FIGURES, exist_ok=True)
os.makedirs(OUTPUT_REPORTS, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'primary': '#2ecc71', 'secondary': '#3498db', 'accent': '#e74c3c', 'warm': '#f39c12'}


def load_trends_data():
    """Load Google Trends data."""
    dfs = []
    for file in os.listdir(TRENDS_DIR):
        if file.endswith('.csv') and file != 'test.csv' and not file.startswith('.'):
            filepath = os.path.join(TRENDS_DIR, file)
            print(f"Loading: {file}")
            df = pd.read_csv(filepath, skiprows=2)
            if 'Week' in df.columns:
                df['Week'] = pd.to_datetime(df['Week'])
                df.set_index('Week', inplace=True)
                dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined
    return pd.DataFrame()


def load_weather_data():
    """Load weather data and resample to weekly."""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('weather_berlin') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No weather data found")

    filepath = os.path.join(DATA_DIR, sorted(files)[-1])
    print(f"Loading weather: {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Resample to weekly on Sunday (to match Google Trends)
    weekly = df.resample('W-SUN').agg({
        'temperature_2m_mean': 'mean',
        'temperature_2m_max': 'max',
        'temperature_2m_min': 'min',
        'precipitation_sum': 'sum',
    })
    weekly.columns = ['temp_mean', 'temp_max', 'temp_min', 'precip_total']
    return weekly


def plot_trends_timeseries(trends_df, save=True):
    """Plot Google Trends time series - standalone."""
    supermarkt_col = [c for c in trends_df.columns if 'supermarkt berlin' in c.lower()]
    wochenmarkt_col = [c for c in trends_df.columns if 'wochenmarkt' in c.lower()]

    if not supermarkt_col:
        print("No supermarkt data found")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Supermarkt Berlin
    ax1 = axes[0]
    ax1.plot(trends_df.index, trends_df[supermarkt_col[0]],
             color=COLORS['primary'], linewidth=1.5)
    ax1.fill_between(trends_df.index, 0, trends_df[supermarkt_col[0]],
                     color=COLORS['primary'], alpha=0.3)

    # Mark Christmas periods
    for year in range(2021, 2027):
        try:
            christmas = pd.Timestamp(f'{year}-12-25')
            ax1.axvline(x=christmas, color='red', alpha=0.4, linestyle='--', linewidth=1)
        except:
            pass

    ax1.set_ylabel('Search Interest (0-100)')
    ax1.set_title('Berlin Supermarket Searches (2021-2026) - Clear Christmas Spikes',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(trends_df.index.min(), trends_df.index.max())

    # Annotate max values
    max_val = trends_df[supermarkt_col[0]].max()
    max_date = trends_df[supermarkt_col[0]].idxmax()
    ax1.annotate(f'Peak: {max_val:.0f}', xy=(max_date, max_val),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=9, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    # Plot 2: Wochenmarkt Berlin
    ax2 = axes[1]
    if wochenmarkt_col:
        ax2.plot(trends_df.index, trends_df[wochenmarkt_col[0]],
                 color=COLORS['secondary'], linewidth=1.5)
        ax2.fill_between(trends_df.index, 0, trends_df[wochenmarkt_col[0]],
                         color=COLORS['secondary'], alpha=0.3)
        ax2.set_title('Berlin Farmers Market Searches - Seasonal Pattern', fontsize=12)
    ax2.set_ylabel('Search Interest (0-100)')
    ax2.set_xlabel('Date')
    ax2.set_xlim(trends_df.index.min(), trends_df.index.max())

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'trends_timeseries.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_weather_correlation(trends_df, weather_df, save=True):
    """Plot weather vs trends correlation."""
    supermarkt_col = [c for c in trends_df.columns if 'supermarkt berlin' in c.lower()]
    if not supermarkt_col:
        return

    # Merge data on matching dates
    combined = trends_df[[supermarkt_col[0]]].copy()
    combined.columns = ['searches']
    combined = combined.join(weather_df[['temp_mean', 'precip_total']], how='inner')
    combined = combined.dropna()

    print(f"Combined data: {len(combined)} weeks")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter 1: Temperature vs Searches
    ax1 = axes[0]

    # Color by month
    combined['month'] = combined.index.month
    scatter = ax1.scatter(combined['temp_mean'], combined['searches'],
                          c=combined['month'], cmap='coolwarm',
                          alpha=0.6, s=40, edgecolor='white', linewidth=0.5)

    # Trend line
    z = np.polyfit(combined['temp_mean'], combined['searches'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(combined['temp_mean'].min(), combined['temp_mean'].max(), 100)
    ax1.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=2,
             label=f'Trend (r={combined["temp_mean"].corr(combined["searches"]):.2f})')

    ax1.set_xlabel('Weekly Mean Temperature (°C)')
    ax1.set_ylabel('Search Interest')
    ax1.set_title('Temperature vs Supermarket Searches', fontsize=12, fontweight='bold')
    ax1.legend()

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Month')

    # Scatter 2: Precipitation vs Searches
    ax2 = axes[1]
    ax2.scatter(combined['precip_total'], combined['searches'],
                c=COLORS['secondary'], alpha=0.6, s=40, edgecolor='white', linewidth=0.5)

    z2 = np.polyfit(combined['precip_total'], combined['searches'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(0, combined['precip_total'].max(), 100)
    ax2.plot(x_line2, p2(x_line2), 'k--', alpha=0.7, linewidth=2,
             label=f'Trend (r={combined["precip_total"].corr(combined["searches"]):.2f})')

    ax2.set_xlabel('Weekly Precipitation (mm)')
    ax2.set_ylabel('Search Interest')
    ax2.set_title('Precipitation vs Supermarket Searches', fontsize=12, fontweight='bold')
    ax2.legend()

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'trends_weather_correlation.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()

    return combined


def plot_monthly_patterns(trends_df, save=True):
    """Plot monthly search patterns."""
    supermarkt_col = [c for c in trends_df.columns if 'supermarkt berlin' in c.lower()]
    wochenmarkt_col = [c for c in trends_df.columns if 'wochenmarkt' in c.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Supermarkt monthly
    if supermarkt_col:
        ax1 = axes[0]
        df = trends_df.copy()
        df['month'] = df.index.month
        monthly = df.groupby('month')[supermarkt_col[0]].agg(['mean', 'std'])

        bars = ax1.bar(range(1, 13), monthly['mean'], yerr=monthly['std'],
                       color=COLORS['primary'], alpha=0.7, capsize=3, edgecolor='white')

        # Highlight December
        bars[11].set_color(COLORS['accent'])

        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(months)
        ax1.set_ylabel('Average Search Interest')
        ax1.set_title('Supermarkt Berlin - Monthly Pattern', fontsize=12, fontweight='bold')
        ax1.axhline(y=monthly['mean'].mean(), color='gray', linestyle='--', alpha=0.7,
                    label=f'Average: {monthly["mean"].mean():.1f}')
        ax1.legend()

    # Wochenmarkt monthly
    if wochenmarkt_col:
        ax2 = axes[1]
        monthly_wm = df.groupby('month')[wochenmarkt_col[0]].agg(['mean', 'std'])

        bars2 = ax2.bar(range(1, 13), monthly_wm['mean'], yerr=monthly_wm['std'],
                        color=COLORS['secondary'], alpha=0.7, capsize=3, edgecolor='white')

        # Highlight summer months
        for i in [4, 5, 6, 7]:  # May-Aug
            bars2[i].set_color(COLORS['warm'])

        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(months)
        ax2.set_ylabel('Average Search Interest')
        ax2.set_title('Wochenmarkt Berlin - Monthly Pattern', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'trends_monthly_patterns.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def main():
    print("=" * 60)
    print("AGRICOM - Trends Analysis (Fixed)")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    trends_df = load_trends_data()
    weather_df = load_weather_data()

    print(f"   Trends: {len(trends_df)} weeks")
    print(f"   Trends date range: {trends_df.index.min()} to {trends_df.index.max()}")
    print(f"   Weather: {len(weather_df)} weeks")
    print(f"   Weather date range: {weather_df.index.min()} to {weather_df.index.max()}")

    # Generate visualizations
    print("\n2. Generating visualizations...")

    print("   - Time series plot...")
    plot_trends_timeseries(trends_df)

    print("   - Weather correlation...")
    plot_weather_correlation(trends_df, weather_df)

    print("   - Monthly patterns...")
    plot_monthly_patterns(trends_df)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
