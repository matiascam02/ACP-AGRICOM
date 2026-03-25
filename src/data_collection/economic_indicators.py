"""
AGRICOM - Economic Indicators Collection
Collects consumer confidence, inflation, and economic data for Germany.

Data Sources:
- OECD API (Consumer Confidence Index)
- Eurostat (HICP Inflation)
- Destatis API (German statistics)

Usage:
    python economic_indicators.py

Output:
    data/raw/economic_indicators_YYYYMMDD.csv
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import os
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw'

# Germany ISO codes
GERMANY_ISO2 = 'DE'
GERMANY_ISO3 = 'DEU'


def fetch_oecd_consumer_confidence() -> pd.DataFrame:
    """
    Fetch Consumer Confidence Index from FRED (OECD source).
    CCI > 100 = optimistic, CCI < 100 = pessimistic.
    Series: CSCICP03DEM665S (Germany, monthly, long-term trend normalised).
    """
    print("Fetching Consumer Confidence Index (FRED/OECD)...")

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CSCICP03DEM665S&cosd=2019-01-01"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(pd.io.common.StringIO(response.text))
        df.columns = ['date', 'consumer_confidence']
        df['date'] = pd.to_datetime(df['date'])
        df['consumer_confidence'] = pd.to_numeric(df['consumer_confidence'], errors='coerce')
        df = df.dropna()
        print(f"  Got {len(df)} months of CCI data")
        return df

    except Exception as e:
        print(f"  FRED CCI failed: {e}")
        return pd.DataFrame()


def fetch_oecd_cpi_inflation() -> pd.DataFrame:
    """
    Fetch monthly CPI inflation from FRED (OECD source) instead of World Bank annual data.
    Series: DEUCPIALLMINMEI (Germany, CPI all items, monthly, index).
    """
    print("Fetching monthly CPI inflation (FRED/OECD)...")

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEUCPIALLMINMEI&cosd=2019-01-01"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(pd.io.common.StringIO(response.text))
        df.columns = ['date', 'cpi_index']
        df['date'] = pd.to_datetime(df['date'])
        df['cpi_index'] = pd.to_numeric(df['cpi_index'], errors='coerce')
        df = df.dropna()

        # Calculate year-over-year inflation rate
        df = df.sort_values('date')
        df['inflation_rate'] = df['cpi_index'].pct_change(12) * 100

        print(f"  Got {len(df)} months of CPI data")
        return df

    except Exception as e:
        print(f"  FRED CPI failed: {e}")
        return pd.DataFrame()


def fetch_destatis_food_prices() -> pd.DataFrame:
    """
    Fetch food price indices from German Federal Statistics Office.
    Uses their Genesis API.
    """
    print("Fetching Destatis food price data...")
    
    # Destatis provides CPI data by category
    # Using their REST API (requires registration for full access)
    
    # Fallback: Use their pre-published data
    # Food and non-alcoholic beverages CPI (2020=100)
    
    # For now, create synthetic based on known patterns
    # In production, register at genesis.destatis.de for API access
    
    print("  ⚠️  Destatis API requires registration. Using Eurostat fallback.")
    return pd.DataFrame()


def fetch_eurostat_hicp() -> pd.DataFrame:
    """
    Fetch Harmonized Index of Consumer Prices from Eurostat.
    Includes food-specific indices.
    """
    print("Fetching Eurostat HICP (food prices)...")
    
    # Eurostat JSON API
    base_url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    
    # HICP - Food and non-alcoholic beverages
    dataset = "prc_hicp_midx"  # Monthly index
    
    params = {
        'format': 'JSON',
        'lang': 'EN',
        'geo': 'DE',
        'coicop': 'CP01',  # Food and non-alcoholic beverages
        'unit': 'I15',  # Index 2015=100
    }
    
    try:
        url = f"{base_url}/{dataset}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse Eurostat JSON format
        time_index = data.get('dimension', {}).get('time', {}).get('category', {}).get('index', {})
        values = data.get('value', {})
        
        rows = []
        for time_key, idx in time_index.items():
            if str(idx) in values:
                rows.append({
                    'date': time_key,
                    'food_price_index': values[str(idx)]
                })
        
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'].str.replace('M', '-') + '-01')
        df = df.sort_values('date')
        print(f"  ✓ Got {len(df)} months of food price data")
        return df
        
    except Exception as e:
        print(f"  ✗ Eurostat API failed: {e}")
        return pd.DataFrame()


def fetch_bundesbank_rates() -> pd.DataFrame:
    """
    Fetch interest rates and economic indicators from Bundesbank.
    """
    print("Fetching Bundesbank data...")
    
    # Bundesbank provides time series via their API
    # Using their Statistics Download format
    
    try:
        # ECB main refinancing rate (affects consumer spending)
        url = "https://api.statistiken.bundesbank.de/rest/data/BBSIS/D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A?format=csv"
        
        response = requests.get(url, timeout=30)
        if response.ok:
            # Parse the CSV
            lines = response.text.strip().split('\n')
            # Find data rows (skip headers)
            data_rows = [l for l in lines if l and not l.startswith('#') and ',' in l]
            if data_rows:
                df = pd.DataFrame([r.split(',') for r in data_rows[1:]], 
                                  columns=['date', 'interest_rate'])
                df['date'] = pd.to_datetime(df['date'])
                df['interest_rate'] = pd.to_numeric(df['interest_rate'], errors='coerce')
                print(f"  ✓ Got interest rate data")
                return df
    except Exception as e:
        print(f"  ✗ Bundesbank API failed: {e}")
    
    return pd.DataFrame()


def combine_indicators(*dataframes) -> pd.DataFrame:
    """Combine all indicators into single DataFrame with monthly frequency."""
    
    # Filter out empty dataframes
    valid_dfs = [df for df in dataframes if not df.empty]
    
    if not valid_dfs:
        print("No indicator data collected!")
        return pd.DataFrame()
    
    # Start with date range
    all_dates = pd.concat([df[['date']] for df in valid_dfs])
    date_range = pd.date_range(
        start=all_dates['date'].min(),
        end=all_dates['date'].max(),
        freq='MS'  # Month start
    )
    
    result = pd.DataFrame({'date': date_range})
    
    for df in valid_dfs:
        # Resample to monthly if needed
        df = df.copy()
        df['month'] = df['date'].dt.to_period('M')
        df_monthly = df.groupby('month').first().reset_index()
        df_monthly['date'] = df_monthly['month'].dt.to_timestamp()
        
        # Merge
        cols_to_merge = [c for c in df_monthly.columns if c not in ['date', 'month']]
        result = result.merge(
            df_monthly[['date'] + cols_to_merge],
            on='date',
            how='left'
        )
    
    # Forward fill missing values
    result = result.ffill()
    
    return result


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived economic features useful for demand forecasting."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Month-over-month changes
    for col in df.columns:
        if col != 'date' and df[col].dtype in ['float64', 'int64']:
            df[f'{col}_mom_change'] = df[col].pct_change() * 100
    
    # Economic sentiment composite (if we have CCI)
    if 'consumer_confidence' in df.columns:
        df['economic_sentiment'] = df['consumer_confidence'].apply(
            lambda x: 'optimistic' if x > 100 else ('pessimistic' if x < 98 else 'neutral')
        )
    
    # Inflation categories
    if 'inflation_rate' in df.columns:
        df['inflation_category'] = pd.cut(
            df['inflation_rate'],
            bins=[-float('inf'), 2, 4, 6, float('inf')],
            labels=['low', 'moderate', 'high', 'very_high']
        )
    
    return df


def main():
    print("=" * 60)
    print("AGRICOM - Economic Indicators Collection")
    print("=" * 60)
    print(f"Country: Germany (DE)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect from various sources
    print("\n1. Collecting data from sources...")
    
    df_cci = fetch_oecd_consumer_confidence()
    df_cpi = fetch_oecd_cpi_inflation()
    df_hicp = fetch_eurostat_hicp()
    df_rates = fetch_bundesbank_rates()

    # Combine all indicators (using OECD monthly CPI instead of World Bank annual)
    print("\n2. Combining indicators...")
    df = combine_indicators(df_cci, df_cpi, df_hicp, df_rates)
    
    if df.empty:
        print("\n⚠️  No data collected. Check network and API availability.")
        return
    
    # Add derived features
    print("\n3. Adding derived features...")
    df = add_derived_features(df)
    
    # Save
    print("\n4. Saving data...")
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"economic_indicators_{timestamp}.csv"
    filepath = OUTPUT_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"   Saved to: {filepath}")
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total months: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if 'consumer_confidence' in df.columns:
        print(f"\nConsumer Confidence (latest): {df['consumer_confidence'].iloc[-1]:.1f}")
    if 'food_price_index' in df.columns:
        print(f"Food Price Index (latest): {df['food_price_index'].iloc[-1]:.1f}")


if __name__ == "__main__":
    main()
