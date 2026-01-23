"""
AGRICOM - Google Trends Chunked Collection
Fetches one keyword at a time with long delays to avoid rate limits.

Usage:
    python google_trends_chunked.py

This script saves progress and can be resumed if interrupted.
"""

import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime
import time
import os
import random
import json

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw/google_trends')
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'progress.json')

# Keywords to collect - one at a time
KEYWORDS = [
    # Shopping behavior (Germany-wide for more data)
    {'keyword': 'bio lebensmittel', 'geo': 'DE', 'name': 'bio_lebensmittel_de'},
    {'keyword': 'bio gemüse', 'geo': 'DE', 'name': 'bio_gemuese_de'},
    {'keyword': 'bio tomaten', 'geo': 'DE', 'name': 'bio_tomaten_de'},
    {'keyword': 'bio obst', 'geo': 'DE', 'name': 'bio_obst_de'},
    {'keyword': 'wochenmarkt', 'geo': 'DE', 'name': 'wochenmarkt_de'},
    {'keyword': 'bauernmarkt', 'geo': 'DE', 'name': 'bauernmarkt_de'},

    # Seasonal/behavior signals
    {'keyword': 'grillen rezepte', 'geo': 'DE', 'name': 'grillen_rezepte_de'},
    {'keyword': 'salat rezepte', 'geo': 'DE', 'name': 'salat_rezepte_de'},
    {'keyword': 'suppe rezepte', 'geo': 'DE', 'name': 'suppe_rezepte_de'},

    # Health/sustainability
    {'keyword': 'vegan essen', 'geo': 'DE', 'name': 'vegan_essen_de'},
    {'keyword': 'nachhaltig einkaufen', 'geo': 'DE', 'name': 'nachhaltig_einkaufen_de'},
]

TIMEFRAME = 'today 5-y'


def load_progress():
    """Load progress from previous run."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'failed': []}


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def fetch_single_keyword(pytrend, keyword, geo, timeframe=TIMEFRAME):
    """Fetch data for a single keyword."""
    try:
        pytrend.build_payload(
            kw_list=[keyword],
            geo=geo,
            timeframe=timeframe
        )
        df = pytrend.interest_over_time()

        if not df.empty and 'isPartial' in df.columns:
            df = df.drop('isPartial', axis=1)

        return df, None
    except Exception as e:
        return None, str(e)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("AGRICOM - Google Trends Chunked Collection")
    print("=" * 60)
    print(f"Keywords to collect: {len(KEYWORDS)}")
    print(f"Timeframe: {TIMEFRAME}")
    print("=" * 60)

    # Load progress
    progress = load_progress()
    print(f"\nPreviously completed: {len(progress['completed'])}")
    print(f"Previously failed: {len(progress['failed'])}")

    # Initialize pytrends
    print("\nInitializing pytrends...")
    time.sleep(random.uniform(3, 7))
    pytrend = TrendReq(hl='de-DE', tz=60, timeout=(10, 30), retries=2, backoff_factor=1)

    for i, kw_config in enumerate(KEYWORDS):
        keyword = kw_config['keyword']
        geo = kw_config['geo']
        name = kw_config['name']

        # Skip if already completed
        if name in progress['completed']:
            print(f"\n[{i+1}/{len(KEYWORDS)}] SKIP (already done): {keyword}")
            continue

        print(f"\n[{i+1}/{len(KEYWORDS)}] Fetching: {keyword} ({geo})")

        # Random delay: 90-180 seconds between requests
        delay = random.uniform(90, 180)
        print(f"   Waiting {delay:.0f}s to avoid rate limit...")
        time.sleep(delay)

        # Fetch data
        df, error = fetch_single_keyword(pytrend, keyword, geo)

        if df is not None and not df.empty:
            # Save individual file
            filepath = os.path.join(OUTPUT_DIR, f'{name}.csv')
            df.to_csv(filepath)
            print(f"   SUCCESS: {len(df)} rows saved to {name}.csv")

            progress['completed'].append(name)
            save_progress(progress)
        else:
            print(f"   FAILED: {error}")
            progress['failed'].append({'name': name, 'error': error})
            save_progress(progress)

            # If rate limited, wait longer
            if '429' in str(error) or 'too many' in str(error).lower():
                print("   Rate limited! Waiting 5 minutes...")
                time.sleep(300)

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Completed: {len(progress['completed'])}")
    print(f"Failed: {len(progress['failed'])}")

    if progress['completed']:
        print("\nSuccessfully collected:")
        for name in progress['completed']:
            print(f"  - {name}")

    if progress['failed']:
        print("\nFailed (try again later):")
        for item in progress['failed']:
            print(f"  - {item['name']}: {item['error'][:50]}...")


if __name__ == "__main__":
    main()
