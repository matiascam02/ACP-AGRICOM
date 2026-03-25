"""
AGRICOM - Master Panel Builder
Constructs the weekly master panel for Phase 2+ analysis.

Index: ISO week (Monday-start), Week 1 2021 -> current week
Sources: Google Trends (4 basket keywords), Weather, Prices (AMI), Economic controls

Output: data/processed/master_panel_YYYYMMDD.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

PROJECT_DIR = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_DIR / 'data' / 'raw'
GT_DIR = RAW_DIR / 'google_trends'
PRICING_DIR = RAW_DIR / 'pricing'
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Basket product mapping: filename stem -> column name
BASKET_PRODUCTS = {
    'bio_tomaten_de': 'gt_bio_tomaten',
    'bio_salat_de': 'gt_bio_salat',
    'bio_gurken_de': 'gt_bio_gurken',
    'bio_paprika_de': 'gt_bio_paprika',
}

# Fallback: if specific product GT not available, use bio_gemuese_de as proxy
PROXY_FILE = 'bio_gemuese_de'


def create_weekly_spine(start='2021-01-04', end=None):
    """Create base panel: one row per ISO week (Monday start)."""
    if end is None:
        # Current Monday
        today = pd.Timestamp.now()
        end = today - pd.Timedelta(days=today.dayofweek)

    spine = pd.DataFrame({
        'week_start': pd.date_range(start=start, end=end, freq='W-MON')
    })
    print(f"Weekly spine: {spine['week_start'].min().date()} to {spine['week_start'].max().date()} ({len(spine)} weeks)")
    return spine


def load_gt_basket():
    """Load 4 basket GT CSVs. Falls back to bio_gemuese_de as proxy for missing products."""
    print("\nLoading Google Trends basket data...")
    gt_frames = {}
    proxy_df = None

    for filename, col_name in BASKET_PRODUCTS.items():
        filepath = GT_DIR / f'{filename}.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Handle both pytrends format (date,keyword) and manual format
            date_col = df.columns[0]
            value_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]

            if date_col != value_col:
                df = df.rename(columns={date_col: 'date', value_col: col_name})
                df = df[['date', col_name]]
            else:
                continue

            df['date'] = pd.to_datetime(df['date'])
            # GT dates are typically Sundays — align to prior Monday
            df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')
            df = df[['week_start', col_name]]
            gt_frames[col_name] = df
            print(f"  Loaded {filename}: {len(df)} weeks")
        else:
            print(f"  MISSING: {filename} — will use proxy")

    # Load proxy if needed
    missing_products = [col for col in BASKET_PRODUCTS.values() if col not in gt_frames]
    if missing_products:
        proxy_path = GT_DIR / f'{PROXY_FILE}.csv'
        if proxy_path.exists():
            pdf = pd.read_csv(proxy_path)
            date_col = pdf.columns[0]
            value_col = pdf.columns[-1]
            pdf = pdf.rename(columns={date_col: 'date', value_col: 'proxy_value'})
            pdf['date'] = pd.to_datetime(pdf['date'])
            pdf['week_start'] = pdf['date'] - pd.to_timedelta(pdf['date'].dt.dayofweek, unit='D')

            for col_name in missing_products:
                proxy_copy = pdf[['week_start', 'proxy_value']].copy()
                proxy_copy = proxy_copy.rename(columns={'proxy_value': col_name})
                gt_frames[col_name] = proxy_copy
                print(f"  PROXY ({PROXY_FILE}) -> {col_name}")
        else:
            print(f"  WARNING: Proxy file {PROXY_FILE} not found. Missing GT columns will be NaN.")

    # Merge all GT series
    if not gt_frames:
        print("  ERROR: No GT data loaded!")
        return pd.DataFrame(columns=['week_start'])

    result = None
    for col_name, df in gt_frames.items():
        if result is None:
            result = df
        else:
            result = result.merge(df, on='week_start', how='outer')

    result = result.sort_values('week_start').reset_index(drop=True)
    return result


def load_weather_weekly():
    """Load latest weather CSV, resample daily -> weekly."""
    print("\nLoading weather data...")

    weather_files = sorted(glob.glob(str(RAW_DIR / 'weather_berlin_*.csv')))
    if not weather_files:
        print("  WARNING: No weather file found!")
        return pd.DataFrame(columns=['week_start'])

    filepath = weather_files[-1]
    df = pd.read_csv(filepath, parse_dates=['time'], index_col='time')
    print(f"  Loaded {filepath}: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")

    # Resample to weekly (Monday start)
    weekly = df.resample('W-MON').agg({
        'temperature_2m_mean': 'mean',
        'sunshine_duration': 'sum',    # seconds -> sum per week
        'precipitation_sum': 'sum',
    })

    # Convert sunshine from seconds to hours
    weekly['sunshine_hours_weekly'] = weekly['sunshine_duration'] / 3600.0
    weekly = weekly.rename(columns={'temperature_2m_mean': 'temp_mean_weekly', 'precipitation_sum': 'precip_sum_weekly'})
    weekly = weekly.drop(columns=['sunshine_duration'])
    weekly = weekly.reset_index().rename(columns={'time': 'week_start'})

    print(f"  Resampled to {len(weekly)} weeks")
    return weekly


def load_price_data():
    """Load interpolated AMI prices if available, else return empty frame."""
    print("\nLoading price data...")

    # Check for interpolated weekly prices
    interpolated_file = PRICING_DIR / 'ami_weekly_interpolated.csv'
    if interpolated_file.exists():
        df = pd.read_csv(interpolated_file, parse_dates=['week_start'])
        print(f"  Loaded interpolated prices: {len(df)} records")

        # Pivot to wide: one premium column per product
        from src.data_collection.price_data_loader import compute_organic_premium
        df = compute_organic_premium(df)

        premium_wide = df.pivot_table(
            index='week_start',
            columns='product',
            values='organic_premium',
            aggfunc='first'
        )
        premium_wide.columns = [f'organic_premium_{c}' for c in premium_wide.columns]
        return premium_wide.reset_index()
    else:
        print("  No AMI price data found. Price columns will be NaN.")
        return pd.DataFrame(columns=['week_start'])


def load_economic_controls():
    """Load economic indicators: CCI and food_price_index."""
    print("\nLoading economic controls...")

    econ_files = sorted(glob.glob(str(RAW_DIR / 'economic_indicators_*.csv')))
    if not econ_files:
        print("  WARNING: No economic indicators file found!")
        return pd.DataFrame(columns=['date'])

    filepath = econ_files[-1]
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"  Loaded {filepath}: {len(df)} months")

    # Select key columns
    cols = ['date']
    if 'consumer_confidence' in df.columns:
        cols.append('consumer_confidence')
        print(f"  CCI range: {df['consumer_confidence'].min():.1f} - {df['consumer_confidence'].max():.1f}")
    else:
        print("  WARNING: consumer_confidence not found")

    if 'food_price_index' in df.columns:
        cols.append('food_price_index')
        print(f"  Food CPI range: {df['food_price_index'].min():.1f} - {df['food_price_index'].max():.1f}")

    if 'inflation_rate' in df.columns:
        cols.append('inflation_rate')

    return df[cols]


def build_panel(spine, gt, weather, prices, econ):
    """Left join all sources onto weekly spine. Forward-fill monthly economic data."""

    panel = spine.copy()

    # Join GT data
    if not gt.empty and 'week_start' in gt.columns:
        panel = panel.merge(gt, on='week_start', how='left')

    # Join weather
    if not weather.empty and 'week_start' in weather.columns:
        panel = panel.merge(weather, on='week_start', how='left')

    # Join prices
    if not prices.empty and 'week_start' in prices.columns:
        panel = panel.merge(prices, on='week_start', how='left')

    # Join economic controls via asof merge (forward-fill monthly to weekly)
    if not econ.empty and 'date' in econ.columns:
        econ_sorted = econ.sort_values('date')
        panel = pd.merge_asof(
            panel.sort_values('week_start'),
            econ_sorted,
            left_on='week_start',
            right_on='date',
            direction='backward'
        )
        panel = panel.drop(columns=['date'], errors='ignore')

    # Add GT normalised columns (0-1 scale)
    for col in BASKET_PRODUCTS.values():
        if col in panel.columns:
            norm_col = col.replace('gt_', 'gt_norm_')
            panel[norm_col] = panel[col] / 100.0

    # Add time features
    panel['week_of_year'] = panel['week_start'].dt.isocalendar().week.astype(int)
    panel['month'] = panel['week_start'].dt.month
    panel['quarter'] = panel['week_start'].dt.quarter
    panel['year'] = panel['week_start'].dt.year

    panel = panel.sort_values('week_start').reset_index(drop=True)
    return panel


def audit_missingness(df):
    """Print missingness report per column."""
    print("\n" + "=" * 60)
    print("MISSINGNESS AUDIT")
    print("=" * 60)

    total_rows = len(df)
    for col in df.columns:
        if col == 'week_start':
            continue
        n_missing = df[col].isna().sum()
        pct = n_missing / total_rows * 100
        status = "OK" if pct < 10 else "WARN" if pct < 50 else "HIGH"
        if n_missing > 0:
            print(f"  [{status}] {col}: {n_missing}/{total_rows} missing ({pct:.1f}%)")
        else:
            print(f"  [ OK ] {col}: complete")


def main():
    print("=" * 60)
    print("AGRICOM - Master Panel Builder")
    print("=" * 60)

    # Step 1: Create weekly spine
    spine = create_weekly_spine()

    # Step 2: Load all data sources
    gt = load_gt_basket()
    weather = load_weather_weekly()
    prices = load_price_data()
    econ = load_economic_controls()

    # Step 3: Build panel
    print("\nBuilding master panel...")
    panel = build_panel(spine, gt, weather, prices, econ)

    # Step 4: Audit
    audit_missingness(panel)

    # Step 5: Save
    timestamp = datetime.now().strftime('%Y%m%d')
    output_path = PROCESSED_DIR / f'master_panel_{timestamp}.csv'
    panel.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Shape: {panel.shape}")
    print(f"Columns: {list(panel.columns)}")
    print(f"Date range: {panel['week_start'].min().date()} to {panel['week_start'].max().date()}")

    # Quick summary stats
    print("\nSample (last 5 weeks):")
    display_cols = [c for c in panel.columns if c.startswith('gt_bio_') and 'norm' not in c]
    if display_cols:
        print(panel[['week_start'] + display_cols].tail().to_string(index=False))

    return panel


if __name__ == "__main__":
    main()
