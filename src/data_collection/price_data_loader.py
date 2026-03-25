"""
AGRICOM - Price Data Loader (AMI Quarterly Fallback — Scenario B)
Loads, validates, and interpolates AMI quarterly organic/conventional prices
to weekly frequency for the demand index.

Usage:
    python price_data_loader.py

If AMI data CSV is not yet uploaded, generates a template CSV for manual entry.

Input:  data/raw/pricing/ami_quarterly_prices_manual.csv
Output: Processed prices integrated into master panel via build_master_panel.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent.parent
RAW_PRICING_DIR = PROJECT_DIR / 'data' / 'raw' / 'pricing'
AMI_FILE = RAW_PRICING_DIR / 'ami_quarterly_prices_manual.csv'
TEMPLATE_FILE = RAW_PRICING_DIR / 'ami_price_template.csv'

EXPECTED_PRODUCTS = ['tomaten', 'salat', 'gurken', 'paprika']


def create_price_template():
    """Generate an empty CSV template for manual AMI data entry."""
    # Quarterly dates from 2021-Q1 to 2025-Q4
    quarters = pd.date_range('2021-01-01', '2025-10-01', freq='QS')

    rows = []
    # Realistic price ranges for German organic produce (EUR/kg)
    price_hints = {
        'tomaten': (3.50, 5.50, 1.80, 2.80),
        'salat': (2.80, 4.50, 1.40, 2.20),
        'gurken': (2.00, 3.50, 1.00, 1.80),
        'paprika': (4.00, 6.50, 2.50, 3.80),
    }

    for q in quarters:
        for product in EXPECTED_PRODUCTS:
            rows.append({
                'date': q.strftime('%Y-%m-%d'),
                'product': product,
                'organic_price_eur_kg': '',
                'conventional_price_eur_kg': '',
            })

    df = pd.DataFrame(rows)

    RAW_PRICING_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEMPLATE_FILE, index=False)
    print(f"Template created: {TEMPLATE_FILE}")
    print(f"\nExpected price ranges (EUR/kg):")
    for product, (org_lo, org_hi, conv_lo, conv_hi) in price_hints.items():
        print(f"  {product}: organic {org_lo}-{org_hi}, conventional {conv_lo}-{conv_hi}")
    print(f"\nFill in prices and save as: {AMI_FILE}")
    return df


def load_ami_quarterly_prices(filepath=None):
    """Load and validate the manually-uploaded AMI quarterly CSV.

    Returns validated DataFrame or None if file not found.
    """
    filepath = Path(filepath) if filepath else AMI_FILE

    if not filepath.exists():
        print(f"AMI price file not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Validate schema
    required_cols = ['date', 'product', 'organic_price_eur_kg', 'conventional_price_eur_kg']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in AMI file: {missing}")

    df['date'] = pd.to_datetime(df['date'])

    # Drop rows with empty prices
    df = df.dropna(subset=['organic_price_eur_kg', 'conventional_price_eur_kg'])
    df['organic_price_eur_kg'] = pd.to_numeric(df['organic_price_eur_kg'], errors='coerce')
    df['conventional_price_eur_kg'] = pd.to_numeric(df['conventional_price_eur_kg'], errors='coerce')
    df = df.dropna(subset=['organic_price_eur_kg', 'conventional_price_eur_kg'])

    # Validate products
    found_products = set(df['product'].unique())
    missing_products = set(EXPECTED_PRODUCTS) - found_products
    if missing_products:
        print(f"WARNING: Missing products in AMI data: {missing_products}")

    # Validate price sanity
    issues = validate_price_data(df)
    if issues:
        print("Data quality warnings:")
        for issue in issues:
            print(f"  - {issue}")

    print(f"Loaded {len(df)} quarterly price records for {len(found_products)} products")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def validate_price_data(df):
    """Check for data quality issues. Returns list of warning strings."""
    issues = []

    # Check organic > conventional
    inverted = df[df['organic_price_eur_kg'] <= df['conventional_price_eur_kg']]
    if len(inverted) > 0:
        issues.append(f"{len(inverted)} rows where organic <= conventional price")

    # Check extreme values
    for col in ['organic_price_eur_kg', 'conventional_price_eur_kg']:
        if df[col].max() > 20:
            issues.append(f"{col} has values > 20 EUR/kg (check units)")
        if df[col].min() < 0.1:
            issues.append(f"{col} has values < 0.10 EUR/kg (check units)")

    # Check temporal coverage
    n_quarters = df.groupby('product')['date'].nunique()
    for product, n in n_quarters.items():
        if n < 8:
            issues.append(f"{product}: only {n} quarters (recommend >= 8 for 2 seasonal cycles)")

    return issues


def interpolate_to_weekly(df):
    """Interpolate quarterly prices to weekly frequency.

    Returns (weekly_df, uncertainty_df).
    """
    weekly_dfs = []

    for product in df['product'].unique():
        product_df = df[df['product'] == product].copy()
        product_df = product_df.set_index('date').sort_index()

        # Resample to weekly (Monday start) and interpolate
        weekly = product_df[['organic_price_eur_kg', 'conventional_price_eur_kg']].resample('W-MON').interpolate(method='linear')

        # Compute interpolation confidence (1.0 at anchor, decaying between)
        anchor_dates = set(product_df.index)
        confidence = []
        for date in weekly.index:
            min_dist = min((abs((date - a).days) for a in anchor_dates), default=90)
            conf = max(0.3, 1.0 - (min_dist / 90.0) * 0.7)
            confidence.append(conf)
        weekly['interpolation_confidence'] = confidence
        weekly['product'] = product
        weekly_dfs.append(weekly.reset_index().rename(columns={'date': 'week_start'}))

    result = pd.concat(weekly_dfs, ignore_index=True)
    return result


def compute_organic_premium(df):
    """Add organic premium column: (organic - conventional) / conventional."""
    df = df.copy()
    df['organic_premium'] = (
        (df['organic_price_eur_kg'] - df['conventional_price_eur_kg'])
        / df['conventional_price_eur_kg']
    )
    return df


def generate_illustrative_prices(food_price_index_series):
    """Generate illustrative price data from food_price_index when AMI data unavailable.

    This is a documented fallback — all outputs must note that prices are synthetic.
    """
    base_prices = {
        'tomaten': {'organic': 4.20, 'conventional': 2.30},
        'salat': {'organic': 3.40, 'conventional': 1.80},
        'gurken': {'organic': 2.60, 'conventional': 1.40},
        'paprika': {'organic': 5.00, 'conventional': 3.10},
    }

    if food_price_index_series is None or food_price_index_series.empty:
        print("WARNING: No food_price_index available for illustrative prices")
        return None

    # Normalise food CPI to 2021-01 baseline
    baseline = food_price_index_series.iloc[0] if len(food_price_index_series) > 0 else 100
    multiplier = food_price_index_series / baseline

    rows = []
    for date, mult in multiplier.items():
        for product, prices in base_prices.items():
            rows.append({
                'date': date,
                'product': product,
                'organic_price_eur_kg': round(prices['organic'] * mult, 2),
                'conventional_price_eur_kg': round(prices['conventional'] * mult, 2),
                'is_synthetic': True,
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("AGRICOM - Price Data Loader (Scenario B: AMI Fallback)")
    print("=" * 60)

    RAW_PRICING_DIR.mkdir(parents=True, exist_ok=True)

    # Try to load AMI data
    df = load_ami_quarterly_prices()

    if df is not None:
        print("\nInterpolating quarterly -> weekly...")
        weekly = interpolate_to_weekly(df)
        weekly = compute_organic_premium(weekly)

        output_path = RAW_PRICING_DIR / 'ami_weekly_interpolated.csv'
        weekly.to_csv(output_path, index=False)
        print(f"Saved weekly prices: {output_path}")
        print(f"Total weekly observations: {len(weekly)}")
    else:
        print("\nNo AMI data found. Creating template...")
        create_price_template()
        print("\nPipeline will continue with NaN prices in master panel.")
        print("H3 price elasticity will use illustrative prices from food_price_index.")


if __name__ == "__main__":
    main()
