"""
AGRICOM - Google Trends Data Collection (Simple Version)
Conservative approach with longer delays to avoid rate limits.

Usage:
    python google_trends_simple.py

Output:
    data/raw/google_trends_berlin_YYYYMMDD.csv
"""

import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime
import time
import os
import random

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')

# Reduced keyword list - most important ones only
KEYWORDS = [
    'bio lebensmittel',      # organic food (most general)
    'bio gemüse',            # organic vegetables
    'bio tomaten',           # organic tomatoes
    'supermarkt berlin',     # supermarket berlin (behavioral proxy)
    'wochenmarkt berlin',    # farmers market berlin
]

# Germany-wide (more data available than Berlin-specific)
GEO = 'DE'  # Germany-wide instead of DE-BE for more reliable data

# Shorter timeframe to reduce data points
TIMEFRAME = 'today 12-m'  # Last 12 months


def fetch_single_keyword(pytrend, keyword, geo=GEO, timeframe=TIMEFRAME):
    """Fetch data for a single keyword with error handling."""
    try:
        pytrend.build_payload(
            kw_list=[keyword],
            geo=geo,
            timeframe=timeframe
        )
        df = pytrend.interest_over_time()

        if not df.empty and 'isPartial' in df.columns:
            df = df.drop('isPartial', axis=1)

        return df
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def main():
    """Main collection routine with conservative rate limiting."""
    print("=" * 60)
    print("AGRICOM - Google Trends (Simple Version)")
    print("=" * 60)
    print(f"Location: Germany ({GEO})")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Keywords: {len(KEYWORDS)}")
    print("=" * 60)

    # Initialize with random delay
    time.sleep(random.uniform(5, 10))
    pytrend = TrendReq(hl='de-DE', tz=60)

    all_data = []

    for i, keyword in enumerate(KEYWORDS):
        print(f"\n[{i+1}/{len(KEYWORDS)}] Fetching: {keyword}")

        # Random delay between 30-90 seconds
        delay = random.uniform(30, 90)
        print(f"    Waiting {delay:.0f}s...")
        time.sleep(delay)

        df = fetch_single_keyword(pytrend, keyword)

        if not df.empty:
            all_data.append(df)
            print(f"    Success: {len(df)} data points")
        else:
            print(f"    No data returned")

    # Combine all data
    if all_data:
        combined = pd.concat(all_data, axis=1)

        # Remove duplicate columns
        combined = combined.loc[:, ~combined.columns.duplicated()]

        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        filepath = os.path.join(OUTPUT_DIR, f'google_trends_germany_{timestamp}.csv')
        combined.to_csv(filepath)

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Saved: {filepath}")
        print(f"Date range: {combined.index.min()} to {combined.index.max()}")
        print(f"Columns: {list(combined.columns)}")
        print("\nSample:")
        print(combined.head())
    else:
        print("\nNo data collected. Google may be rate limiting.")
        print("Try again later or use Google Trends web interface manually.")


if __name__ == "__main__":
    main()
