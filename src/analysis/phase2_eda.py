"""
AGRICOM - Phase 2 Exploratory Data Analysis
Produces annotated GT time series, correlation heatmap,
seasonal decomposition, and descriptive statistics.

Input:  data/processed/master_panel_*.csv
Output: outputs/figures/ and outputs/tables/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
FIGURE_DIR = PROJECT_DIR / 'outputs' / 'figures'
TABLE_DIR = PROJECT_DIR / 'outputs' / 'tables'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

PRODUCT_CONFIG = {
    'gt_bio_tomaten': {'label': 'Bio Tomaten', 'color': '#e74c3c'},
    'gt_bio_salat': {'label': 'Bio Salat', 'color': '#2ecc71'},
    'gt_bio_gurken': {'label': 'Bio Gurken', 'color': '#27ae60'},
    'gt_bio_paprika': {'label': 'Bio Paprika', 'color': '#f39c12'},
}

EVENTS = [
    {'date': '2021-06-01', 'label': 'COVID restrictions ease'},
    {'date': '2022-02-24', 'label': 'Russia-Ukraine war'},
    {'date': '2022-07-01', 'label': 'Inflation spike'},
    {'date': '2022-09-08', 'label': 'ECB rate hikes begin'},
    {'date': '2023-06-15', 'label': 'Summer peak 2023'},
    {'date': '2024-06-15', 'label': 'Summer peak 2024'},
    {'date': '2025-06-15', 'label': 'Summer peak 2025'},
]


def load_master_panel():
    """Load the latest master panel CSV."""
    files = sorted(glob.glob(str(PROCESSED_DIR / 'master_panel_*.csv')))
    if not files:
        raise FileNotFoundError("No master_panel_*.csv found in data/processed/")
    df = pd.read_csv(files[-1], parse_dates=['week_start'])
    print(f"Loaded panel: {len(df)} weeks, {len(df.columns)} columns")
    return df


def plot_annotated_gt_timeseries(df):
    """4-panel figure: one subplot per product with event annotations."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Google Trends Search Interest — Organic Produce (Germany)', fontsize=14, fontweight='bold')

    product_cols = list(PRODUCT_CONFIG.keys())

    for ax, col in zip(axes, product_cols):
        config = PRODUCT_CONFIG[col]
        series = df.dropna(subset=[col])

        ax.plot(series['week_start'], series[col], color=config['color'], linewidth=1.2, alpha=0.8)

        # Rolling average
        if len(series) > 8:
            rolling = series[col].rolling(8, center=True).mean()
            ax.plot(series['week_start'], rolling, color=config['color'], linewidth=2, alpha=0.5, linestyle='--')

        ax.set_ylabel('Search Interest')
        ax.set_title(config['label'], fontsize=11, fontweight='bold', loc='left')
        ax.set_ylim(0, 105)

        # Event annotations
        for event in EVENTS:
            event_date = pd.Timestamp(event['date'])
            if event_date >= series['week_start'].min() and event_date <= series['week_start'].max():
                ax.axvline(event_date, color='gray', linestyle=':', alpha=0.6, linewidth=0.8)
                ax.text(event_date, ax.get_ylim()[1] * 0.92, event['label'],
                       rotation=45, fontsize=7, ha='left', va='top', color='gray')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    filepath = FIGURE_DIR / 'gt_timeseries_annotated.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_correlation_heatmap(df):
    """Pearson correlation matrix heatmap across driver variables."""
    # Select numeric driver columns
    driver_cols = []
    for col in df.columns:
        if col.startswith('gt_bio_') and 'norm' not in col:
            driver_cols.append(col)
    driver_cols.extend(['temp_mean_weekly', 'sunshine_hours_weekly', 'precip_sum_weekly'])
    if 'consumer_confidence' in df.columns:
        driver_cols.append('consumer_confidence')
    if 'food_price_index' in df.columns:
        driver_cols.append('food_price_index')

    available = [c for c in driver_cols if c in df.columns]
    corr_data = df[available].dropna()
    corr_matrix = corr_data.corr()

    # Compute p-values
    n = len(corr_data)
    p_matrix = pd.DataFrame(np.zeros_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns)
    for i, col_i in enumerate(corr_matrix.columns):
        for j, col_j in enumerate(corr_matrix.columns):
            if i != j:
                _, p = stats.pearsonr(corr_data[col_i], corr_data[col_j])
                p_matrix.iloc[i, j] = p

    # Create annotation with significance stars
    annot = corr_matrix.round(2).astype(str)
    for i in range(len(annot)):
        for j in range(len(annot.columns)):
            p = p_matrix.iloc[i, j]
            if i != j:
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                annot.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}{stars}"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=annot, fmt='', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Pearson r'})
    ax.set_title('Correlation Matrix — Demand Driver Variables\n(* p<0.05, ** p<0.01, *** p<0.001)',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    filepath = FIGURE_DIR / 'correlation_matrix.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_seasonal_decomposition(df):
    """Seasonal decomposition per product (additive, period=52)."""
    product_cols = list(PRODUCT_CONFIG.keys())

    for col in product_cols:
        config = PRODUCT_CONFIG[col]
        series = df.set_index('week_start')[col].dropna()

        if len(series) < 104:  # Need at least 2 full years
            print(f"  Skipping {col}: insufficient data ({len(series)} < 104 weeks)")
            continue

        # Interpolate gaps for decomposition
        series = series.interpolate(method='linear')

        try:
            decomp = seasonal_decompose(series, model='additive', period=52)

            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            fig.suptitle(f'Seasonal Decomposition — {config["label"]}', fontsize=13, fontweight='bold')

            axes[0].plot(decomp.observed, color=config['color'], linewidth=1)
            axes[0].set_ylabel('Observed')

            axes[1].plot(decomp.trend, color='navy', linewidth=1.5)
            axes[1].set_ylabel('Trend')

            axes[2].plot(decomp.seasonal, color='green', linewidth=1)
            axes[2].set_ylabel('Seasonal')

            axes[3].plot(decomp.resid, color='gray', linewidth=0.8, alpha=0.7)
            axes[3].axhline(0, color='black', linewidth=0.5)
            axes[3].set_ylabel('Residual')

            plt.tight_layout()
            product_name = col.replace('gt_bio_', '')
            filepath = FIGURE_DIR / f'seasonal_decomposition_{product_name}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"  Decomposition failed for {col}: {e}")

    # Summary table: peak/trough week, seasonal amplitude
    summary_rows = []
    for col in product_cols:
        series = df.set_index('week_start')[col].dropna().interpolate()
        if len(series) < 104:
            continue
        try:
            decomp = seasonal_decompose(series, model='additive', period=52)
            seasonal = decomp.seasonal.iloc[:52]
            peak_week = seasonal.idxmax()
            trough_week = seasonal.idxmin()
            amplitude = seasonal.max() - seasonal.min()
            summary_rows.append({
                'product': col.replace('gt_bio_', ''),
                'peak_week': peak_week.isocalendar()[1] if hasattr(peak_week, 'isocalendar') else 'N/A',
                'trough_week': trough_week.isocalendar()[1] if hasattr(trough_week, 'isocalendar') else 'N/A',
                'seasonal_amplitude': round(amplitude, 2),
            })
        except Exception:
            pass

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        filepath = TABLE_DIR / 'seasonal_summary.csv'
        summary_df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")


def generate_descriptive_stats(df):
    """Descriptive statistics table for key numeric columns."""
    numeric_cols = [c for c in df.columns if c not in ['week_start', 'week_of_year', 'month', 'quarter', 'year']
                    and df[c].dtype in ['float64', 'int64']]

    desc = df[numeric_cols].describe().T
    desc['skewness'] = df[numeric_cols].skew()
    desc['kurtosis'] = df[numeric_cols].kurtosis()
    desc = desc.round(3)

    filepath = TABLE_DIR / 'descriptive_statistics.csv'
    desc.to_csv(filepath)
    print(f"Saved: {filepath}")
    print(f"\n{desc.to_string()}")
    return desc


def main():
    print("=" * 60)
    print("AGRICOM - Phase 2 Exploratory Data Analysis")
    print("=" * 60)

    df = load_master_panel()

    print("\n--- 2a. Annotated GT Time-Series ---")
    plot_annotated_gt_timeseries(df)

    print("\n--- 2b. Correlation Matrix ---")
    plot_correlation_heatmap(df)

    print("\n--- 2c. Seasonal Decomposition ---")
    plot_seasonal_decomposition(df)

    print("\n--- 2d. Descriptive Statistics ---")
    generate_descriptive_stats(df)

    print("\n" + "=" * 60)
    print("Phase 2 EDA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
