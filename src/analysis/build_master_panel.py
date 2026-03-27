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
    """Load 4 basket GT CSVs (monthly data from Google Trends).
    
    Each CSV has columns: "Time" (monthly date), "<keyword>" (interest 0-100).
    Monthly values are forward-filled to weekly granularity via merge_asof.
    No proxy fallback — all 4 product files are required.
    """
    print("\nLoading Google Trends basket data...")
    gt_frames = {}

    for filename, col_name in BASKET_PRODUCTS.items():
        filepath = GT_DIR / f'{filename}.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            # Identify date and value columns (handle "Time" or "date" header)
            date_col = df.columns[0]
            value_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]

            if date_col == value_col:
                print(f"  SKIP {filename}: single column")
                continue

            df = df.rename(columns={date_col: 'date', value_col: col_name})
            df = df[['date', col_name]].dropna(subset=['date'])
            df['date'] = pd.to_datetime(df['date'])
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

            # Detect granularity: if median gap > 20 days, treat as monthly
            if len(df) > 2:
                median_gap = df['date'].diff().median().days
            else:
                median_gap = 30

            if median_gap > 20:
                # Monthly data — keep as-is; will forward-fill to weekly later
                df = df.rename(columns={'date': 'month_start'})
                df = df.sort_values('month_start').reset_index(drop=True)
                gt_frames[col_name] = ('monthly', df)
                print(f"  Loaded {filename}: {len(df)} months (monthly, will forward-fill to weekly)")
            else:
                # Weekly data — align to Monday
                df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')
                df = df[['week_start', col_name]]
                gt_frames[col_name] = ('weekly', df)
                print(f"  Loaded {filename}: {len(df)} weeks")
        else:
            print(f"  MISSING: {filename} — column will be NaN")

    if not gt_frames:
        print("  ERROR: No GT data loaded!")
        return pd.DataFrame(columns=['week_start'])

    # Build a unified weekly GT frame
    # Start with a full weekly spine covering all GT dates
    all_dates = []
    for col_name, (granularity, df) in gt_frames.items():
        date_col = 'month_start' if granularity == 'monthly' else 'week_start'
        all_dates.extend(df[date_col].tolist())
    date_min = min(all_dates)
    date_max = max(all_dates)
    # Extend to cover the last month if monthly
    gt_spine = pd.DataFrame({
        'week_start': pd.date_range(
            start=date_min - pd.Timedelta(days=date_min.dayofweek),
            end=date_max + pd.Timedelta(days=31),
            freq='W-MON'
        )
    })

    for col_name, (granularity, df) in gt_frames.items():
        if granularity == 'monthly':
            # Forward-fill monthly values onto weekly spine via merge_asof
            df_sorted = df.sort_values('month_start')
            gt_spine = pd.merge_asof(
                gt_spine.sort_values('week_start'),
                df_sorted,
                left_on='week_start',
                right_on='month_start',
                direction='backward'
            )
            gt_spine = gt_spine.drop(columns=['month_start'], errors='ignore')
        else:
            # Weekly — direct merge
            gt_spine = gt_spine.merge(df, on='week_start', how='left')

    gt_spine = gt_spine.sort_values('week_start').reset_index(drop=True)
    return gt_spine


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
