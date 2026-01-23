"""
AGRICOM - Google Trends Data Collection
Collects search interest data for organic produce terms in Berlin.

Usage:
    python google_trends.py

Output:
    data/raw/google_trends_berlin_YYYYMMDD.csv
"""

import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import os

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')

# Keywords to track - German and English variations
KEYWORDS = {
    'organic_general': [
        'bio lebensmittel',      # organic food
        'bio gemüse',            # organic vegetables
        'organic food berlin',
        'bio supermarkt',        # organic supermarket
    ],
    'specific_produce': [
        'bio tomaten',           # organic tomatoes
        'bio salat',             # organic salad/lettuce
        'bio obst',              # organic fruit
        'bio äpfel',             # organic apples
        'bio beeren',            # organic berries
    ],
    'sustainability': [
        'nachhaltig einkaufen',  # sustainable shopping
        'regional lebensmittel', # regional food
        'saisonal kochen',       # seasonal cooking
    ],
    'behavioral': [
        'supermarkt berlin',     # supermarket berlin
        'einkaufen berlin',      # shopping berlin
        'wochenmarkt berlin',    # farmers market berlin
    ]
}

# Berlin geo code
GEO = 'DE-BE'

# Timeframe: last 2 years
TIMEFRAME = 'today 5-y'  # Last 5 years for better seasonality analysis


def initialize_pytrends():
    """Initialize pytrends with appropriate settings."""
    return TrendReq(
        hl='de-DE',  # German language
        tz=60,       # Berlin timezone (UTC+1)
        timeout=(10, 25),
        retries=3,
        backoff_factor=0.5
    )


def fetch_interest_over_time(pytrend, keywords, geo=GEO, timeframe=TIMEFRAME):
    """
    Fetch interest over time for a list of keywords.

    Args:
        pytrend: TrendReq instance
        keywords: List of keywords (max 5)
        geo: Geographic location code
        timeframe: Time range string

    Returns:
        DataFrame with interest over time
    """
    if len(keywords) > 5:
        raise ValueError("Maximum 5 keywords per request")

    pytrend.build_payload(
        kw_list=keywords,
        geo=geo,
        timeframe=timeframe
    )

    df = pytrend.interest_over_time()

    if not df.empty and 'isPartial' in df.columns:
        df = df.drop('isPartial', axis=1)

    return df


def fetch_related_queries(pytrend, keyword, geo=GEO, timeframe=TIMEFRAME):
    """
    Fetch related queries for a keyword.

    Args:
        pytrend: TrendReq instance
        keyword: Single keyword
        geo: Geographic location code
        timeframe: Time range string

    Returns:
        Dictionary with 'top' and 'rising' DataFrames
    """
    pytrend.build_payload(
        kw_list=[keyword],
        geo=geo,
        timeframe=timeframe
    )

    return pytrend.related_queries()


def collect_all_keywords(pytrend, keywords_dict):
    """
    Collect data for all keyword categories.

    Args:
        pytrend: TrendReq instance
        keywords_dict: Dictionary of category -> keyword list

    Returns:
        DataFrame with all keywords
    """
    all_data = []

    for category, keywords in keywords_dict.items():
        print(f"\nCollecting: {category}")

        # Process in batches of 5
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            print(f"  Batch: {batch}")

            try:
                df = fetch_interest_over_time(pytrend, batch)

                if not df.empty:
                    df['category'] = category
                    all_data.append(df)
                    print(f"  Success: {len(df)} rows")
                else:
                    print(f"  Warning: No data returned")

            except Exception as e:
                print(f"  Error: {e}")

            # Rate limiting - wait 60 seconds between requests
            print("  Waiting 60s for rate limit...")
            time.sleep(60)

    if all_data:
        return pd.concat(all_data, axis=1)
    return pd.DataFrame()


def save_data(df, filename_prefix='google_trends_berlin'):
    """Save data to CSV with timestamp."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{filename_prefix}_{timestamp}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(filepath)
    print(f"\nSaved to: {filepath}")
    return filepath


def main():
    """Main collection routine."""
    print("=" * 60)
    print("AGRICOM - Google Trends Data Collection")
    print("=" * 60)
    print(f"Location: Berlin (DE-BE)")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Keywords: {sum(len(v) for v in KEYWORDS.values())} total")
    print("=" * 60)

    # Initialize
    pytrend = initialize_pytrends()

    # Collect data
    print("\nStarting collection (this will take several minutes due to rate limits)...")
    df = collect_all_keywords(pytrend, KEYWORDS)

    if not df.empty:
        # Save raw data
        filepath = save_data(df)

        # Print summary
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print("\nSample data:")
        print(df.head())
    else:
        print("\nNo data collected. Check API access and keywords.")


if __name__ == "__main__":
    main()
