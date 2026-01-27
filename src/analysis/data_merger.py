"""
AGRICOM - Data Merger
Combines all data sources into a unified analysis-ready dataset.

Usage:
    python data_merger.py

Output:
    data/processed/agricom_unified_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import glob
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'


def load_weather_data() -> pd.DataFrame:
    """Load and prepare weather data."""
    print("Loading weather data...")
    
    weather_files = list(RAW_DIR.glob('weather_berlin_*.csv'))
    if not weather_files:
        print("  ⚠️  No weather data found")
        return pd.DataFrame()
    
    # Use most recent file
    latest_file = max(weather_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file, parse_dates=['time'])
    df = df.rename(columns={'time': 'date'})
    
    # Select key columns
    weather_cols = [
        'date',
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
        'precipitation_sum', 'rain_sum',
        'sunshine_duration',
        'temp_anomaly', 'temp_anomaly_category',
        'is_rainy', 'is_hot', 'is_cold',
        'season', 'is_weekend'
    ]
    
    available_cols = ['date'] + [c for c in weather_cols if c in df.columns]
    df = df[available_cols]
    
    print(f"  ✓ Loaded {len(df)} days of weather data")
    return df


def load_events_data() -> pd.DataFrame:
    """Load and prepare events data."""
    print("Loading events data...")
    
    events_files = list(RAW_DIR.glob('events_berlin_*.csv'))
    if not events_files:
        print("  ⚠️  No events data found")
        return pd.DataFrame()
    
    latest_file = max(events_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file, parse_dates=['date'])
    
    # Create event indicators by date
    event_summary = df.groupby('date').agg({
        'event_type': lambda x: ','.join(x.unique()) if len(x) > 0 else '',
        'name': 'count'  # Number of events
    }).reset_index()
    event_summary = event_summary.rename(columns={'name': 'event_count'})
    
    # Add binary flags
    event_summary['has_bundesliga'] = event_summary['event_type'].str.contains('Bundesliga', na=False)
    event_summary['has_holiday'] = event_summary['event_type'].str.contains('holiday', case=False, na=False)
    
    print(f"  ✓ Loaded {len(event_summary)} days with events")
    return event_summary


def load_gdelt_data() -> pd.DataFrame:
    """Load and prepare GDELT sentiment data."""
    print("Loading GDELT sentiment data...")
    
    # Try timeline data first (more granular)
    timeline_files = list(RAW_DIR.glob('gdelt_timeline_*.csv'))
    
    if timeline_files:
        latest_file = max(timeline_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file, parse_dates=['datetime'])
        df['date'] = df['datetime'].dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        # Aggregate by date
        gdelt_daily = df.groupby('date').agg({
            'value': ['mean', 'std', 'count']  # Sentiment metrics
        }).reset_index()
        gdelt_daily.columns = ['date', 'gdelt_sentiment_mean', 'gdelt_sentiment_std', 'gdelt_article_count']
        
        print(f"  ✓ Loaded {len(gdelt_daily)} days of GDELT data")
        return gdelt_daily
    
    print("  ⚠️  No GDELT data found")
    return pd.DataFrame()


def load_trends_data() -> pd.DataFrame:
    """Load and prepare Google Trends data."""
    print("Loading Google Trends data...")
    
    trends_dir = RAW_DIR / 'google_trends'
    if not trends_dir.exists():
        print("  ⚠️  No trends directory found")
        return pd.DataFrame()
    
    all_trends = []
    
    for csv_file in trends_dir.glob('*.csv'):
        if csv_file.name in ['progress.json', 'all_trends_combined.csv', 'DOWNLOAD_GUIDE.md']:
            continue
            
        try:
            df = pd.read_csv(csv_file, skiprows=1)
            if len(df.columns) >= 2:
                date_col = df.columns[0]
                value_col = df.columns[1]
                
                df = df.rename(columns={date_col: 'date', value_col: 'value'})
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                
                # Extract keyword from filename
                keyword = csv_file.stem.replace('trends_', '').replace('_de_5y', '').replace('_', ' ')
                df['keyword'] = keyword
                
                all_trends.append(df[['date', 'value', 'keyword']])
        except Exception as e:
            continue
    
    if not all_trends:
        print("  ⚠️  No valid trends files found")
        return pd.DataFrame()
    
    # Combine and pivot
    df_trends = pd.concat(all_trends, ignore_index=True)
    
    # Pivot to wide format
    df_wide = df_trends.pivot_table(
        index='date',
        columns='keyword',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Rename columns with prefix
    df_wide.columns = ['date'] + [f'trends_{c.replace(" ", "_")}' for c in df_wide.columns[1:]]
    
    # Create composite trends index
    trend_cols = [c for c in df_wide.columns if c.startswith('trends_')]
    if trend_cols:
        df_wide['trends_composite'] = df_wide[trend_cols].mean(axis=1)
    
    print(f"  ✓ Loaded {len(df_wide)} weeks of trends data ({len(trend_cols)} keywords)")
    return df_wide


def load_economic_data() -> pd.DataFrame:
    """Load economic indicators."""
    print("Loading economic indicators...")
    
    econ_files = list(RAW_DIR.glob('economic_indicators_*.csv'))
    if not econ_files:
        print("  ⚠️  No economic data found")
        return pd.DataFrame()
    
    latest_file = max(econ_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file, parse_dates=['date'])
    
    print(f"  ✓ Loaded {len(df)} months of economic data")
    return df


def create_unified_dataset(
    df_weather: pd.DataFrame,
    df_events: pd.DataFrame,
    df_gdelt: pd.DataFrame,
    df_trends: pd.DataFrame,
    df_economic: pd.DataFrame,
    frequency: str = 'D'  # Daily by default
) -> pd.DataFrame:
    """
    Merge all data sources into unified dataset.
    
    Args:
        frequency: 'D' for daily, 'W' for weekly, 'M' for monthly
    """
    print(f"\nCreating unified dataset (frequency: {frequency})...")
    
    # Determine date range from available data
    all_dates = []
    for df in [df_weather, df_events, df_gdelt, df_trends]:
        if not df.empty and 'date' in df.columns:
            all_dates.extend(df['date'].dropna().tolist())
    
    if not all_dates:
        print("  ⚠️  No date data available")
        return pd.DataFrame()
    
    # Create base date range
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    date_range = pd.date_range(start=min_date, end=max_date, freq=frequency)
    unified = pd.DataFrame({'date': date_range})
    
    # Merge weather (daily)
    if not df_weather.empty:
        if frequency == 'D':
            unified = unified.merge(df_weather, on='date', how='left')
        else:
            # Aggregate weather to target frequency
            df_weather_agg = df_weather.set_index('date').resample(frequency).agg({
                'temperature_2m_mean': 'mean',
                'precipitation_sum': 'sum',
                'is_rainy': 'sum',  # Count of rainy days
                'is_hot': 'sum',
                'is_cold': 'sum',
            }).reset_index()
            unified = unified.merge(df_weather_agg, on='date', how='left')
    
    # Merge events
    if not df_events.empty:
        if frequency == 'D':
            unified = unified.merge(df_events, on='date', how='left')
        else:
            df_events_agg = df_events.set_index('date').resample(frequency).agg({
                'event_count': 'sum',
                'has_bundesliga': 'sum',
                'has_holiday': 'sum',
            }).reset_index()
            unified = unified.merge(df_events_agg, on='date', how='left')
    
    # Merge GDELT
    if not df_gdelt.empty:
        if frequency == 'D':
            unified = unified.merge(df_gdelt, on='date', how='left')
        else:
            df_gdelt_agg = df_gdelt.set_index('date').resample(frequency).agg({
                'gdelt_sentiment_mean': 'mean',
                'gdelt_article_count': 'sum',
            }).reset_index()
            unified = unified.merge(df_gdelt_agg, on='date', how='left')
    
    # Merge trends (already weekly)
    if not df_trends.empty:
        # Trends data is weekly, need to align
        df_trends_indexed = df_trends.set_index('date')
        
        if frequency == 'D':
            # Forward fill weekly trends to daily
            df_trends_daily = df_trends_indexed.resample('D').ffill().reset_index()
            unified = unified.merge(df_trends_daily, on='date', how='left')
        else:
            df_trends_agg = df_trends_indexed.resample(frequency).mean().reset_index()
            unified = unified.merge(df_trends_agg, on='date', how='left')
    
    # Merge economic (monthly)
    if not df_economic.empty:
        df_economic_indexed = df_economic.set_index('date')
        
        if frequency in ['D', 'W']:
            # Forward fill monthly to target frequency
            df_economic_filled = df_economic_indexed.resample(frequency).ffill().reset_index()
            unified = unified.merge(df_economic_filled, on='date', how='left')
        else:
            unified = unified.merge(df_economic, on='date', how='left')
    
    # Fill missing values
    unified = unified.ffill().bfill()
    
    print(f"  ✓ Unified dataset: {len(unified)} rows, {len(unified.columns)} columns")
    return unified


def add_lag_features(df: pd.DataFrame, columns: list, lags: list = [7, 14, 30]) -> pd.DataFrame:
    """Add lagged features for time series analysis."""
    print("Adding lag features...")
    
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame, columns: list, windows: list = [7, 14, 30]) -> pd.DataFrame:
    """Add rolling window features."""
    print("Adding rolling features...")
    
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            df[f'{col}_roll{window}_mean'] = df[col].rolling(window=window).mean()
            df[f'{col}_roll{window}_std'] = df[col].rolling(window=window).std()
    
    return df


def main():
    print("=" * 60)
    print("AGRICOM - Data Merger")
    print("=" * 60)
    
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all data sources
    print("\n1. Loading data sources...")
    df_weather = load_weather_data()
    df_events = load_events_data()
    df_gdelt = load_gdelt_data()
    df_trends = load_trends_data()
    df_economic = load_economic_data()
    
    # Create unified dataset (weekly for forecasting)
    print("\n2. Creating weekly unified dataset...")
    df_weekly = create_unified_dataset(
        df_weather, df_events, df_gdelt, df_trends, df_economic,
        frequency='W'
    )
    
    if df_weekly.empty:
        print("\n⚠️  No data to merge!")
        return
    
    # Add lag features
    numeric_cols = df_weekly.select_dtypes(include=[np.number]).columns.tolist()
    key_cols = [c for c in numeric_cols if any(x in c for x in ['temp', 'precip', 'trends', 'sentiment'])]
    
    df_weekly = add_lag_features(df_weekly, key_cols[:5], lags=[1, 2, 4])  # 1, 2, 4 weeks
    df_weekly = add_rolling_features(df_weekly, key_cols[:3], windows=[4, 8])  # 4, 8 week windows
    
    # Save
    print("\n3. Saving processed data...")
    
    output_file = PROCESSED_DIR / 'agricom_unified_weekly.csv'
    df_weekly.to_csv(output_file, index=False)
    print(f"   Saved: {output_file}")
    
    # Also create daily version
    print("\n4. Creating daily unified dataset...")
    df_daily = create_unified_dataset(
        df_weather, df_events, df_gdelt, df_trends, df_economic,
        frequency='D'
    )
    
    output_file_daily = PROCESSED_DIR / 'agricom_unified_daily.csv'
    df_daily.to_csv(output_file_daily, index=False)
    print(f"   Saved: {output_file_daily}")
    
    # Summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nWeekly dataset:")
    print(f"  Rows: {len(df_weekly)}")
    print(f"  Columns: {len(df_weekly.columns)}")
    print(f"  Date range: {df_weekly['date'].min()} to {df_weekly['date'].max()}")
    print(f"\nDaily dataset:")
    print(f"  Rows: {len(df_daily)}")
    print(f"  Columns: {len(df_daily.columns)}")
    
    print("\nKey columns available:")
    for col in df_weekly.columns[:20]:
        print(f"  - {col}")
    if len(df_weekly.columns) > 20:
        print(f"  ... and {len(df_weekly.columns) - 20} more")


if __name__ == "__main__":
    main()
