"""
AGRICOM - Google Trends Basket Keywords Collection
Collects the 4 product-specific organic produce keywords for the demand index.

Keywords: bio tomaten, bio salat, bio gurken, bio paprika
Geography: Germany-wide (DE)
Timeframe: 5 years (weekly)

Usage:
    python collect_gt_basket.py

This script saves progress and can be resumed if interrupted.
Reuses existing bio_tomaten_de.csv if present and valid.
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
MANUAL_DIR = os.path.join(OUTPUT_DIR, 'manual')
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'basket_progress.json')

# The 4 basket keywords for the demand index
BASKET_KEYWORDS = [
    {'keyword': 'bio tomaten', 'geo': 'DE', 'name': 'bio_tomaten_de'},
    {'keyword': 'bio salat', 'geo': 'DE', 'name': 'bio_salat_de'},
    {'keyword': 'bio gurken', 'geo': 'DE', 'name': 'bio_gurken_de'},
    {'keyword': 'bio paprika', 'geo': 'DE', 'name': 'bio_paprika_de'},
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


def validate_existing_file(name):
    """Check if a keyword CSV already exists and has valid data (200+ weeks)."""
    filepath = os.path.join(OUTPUT_DIR, f'{name}.csv')
    if not os.path.exists(filepath):
        return False

    try:
        df = pd.read_csv(filepath)
        if len(df) >= 200:
            print(f"  Found valid existing file: {name}.csv ({len(df)} rows)")
            return True
    except Exception:
        pass
    return False


def fetch_single_keyword(pytrend, keyword, geo, timeframe=TIMEFRAME):
    """Fetch data for a single keyword. Returns (df, error)."""
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


def print_manual_instructions(remaining_keywords):
    """Print instructions for manual download if pytrends fails."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("pytrends was rate-limited. Please download manually:")
    print()
    for kw in remaining_keywords:
        keyword = kw['keyword']
        name = kw['name']
        url = f"https://trends.google.com/trends/explore?date=today%205-y&geo=DE&q={keyword.replace(' ', '%20')}"
        print(f"  {keyword}:")
        print(f"    URL: {url}")
        print(f"    Save as: data/raw/google_trends/manual/{name}_manual.csv")
        print()
    print(f"After downloading, place files in: {MANUAL_DIR}")
    print("Then re-run this script or proceed to build_master_panel.py")
    print("=" * 60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MANUAL_DIR, exist_ok=True)

    print("=" * 60)
    print("AGRICOM - Google Trends Basket Collection")
    print("=" * 60)
    print(f"Basket keywords: {len(BASKET_KEYWORDS)}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Geography: Germany (DE)")
    print("=" * 60)

    progress = load_progress()
    print(f"\nPreviously completed: {len(progress['completed'])}")

    # Check for existing valid files
    for kw_config in BASKET_KEYWORDS:
        name = kw_config['name']
        if name not in progress['completed'] and validate_existing_file(name):
            progress['completed'].append(name)
            save_progress(progress)

    # Determine remaining keywords
    remaining = [kw for kw in BASKET_KEYWORDS if kw['name'] not in progress['completed']]

    if not remaining:
        print("\nAll 4 basket keywords already collected!")
        print("Files:")
        for kw in BASKET_KEYWORDS:
            filepath = os.path.join(OUTPUT_DIR, f"{kw['name']}.csv")
            print(f"  {filepath}")
        return

    print(f"\nRemaining to collect: {len(remaining)}")
    for kw in remaining:
        print(f"  - {kw['keyword']}")

    # Initialize pytrends
    print("\nInitializing pytrends...")
    time.sleep(random.uniform(3, 7))
    pytrend = TrendReq(hl='de-DE', tz=60, timeout=(10, 30), retries=2, backoff_factor=1)

    rate_limited = False
    for i, kw_config in enumerate(remaining):
        keyword = kw_config['keyword']
        geo = kw_config['geo']
        name = kw_config['name']

        print(f"\n[{i+1}/{len(remaining)}] Fetching: {keyword} ({geo})")

        # Delay between requests
        delay = random.uniform(90, 180)
        print(f"  Waiting {delay:.0f}s to avoid rate limit...")
        time.sleep(delay)

        df, error = fetch_single_keyword(pytrend, keyword, geo)

        if df is not None and not df.empty:
            # Save with clean format: date,keyword_name
            filepath = os.path.join(OUTPUT_DIR, f'{name}.csv')
            df.to_csv(filepath)
            print(f"  SUCCESS: {len(df)} rows saved to {name}.csv")

            progress['completed'].append(name)
            save_progress(progress)
        else:
            print(f"  FAILED: {error}")
            progress['failed'].append({'name': name, 'error': error, 'timestamp': datetime.now().isoformat()})
            save_progress(progress)

            if '429' in str(error) or 'too many' in str(error).lower():
                print("  Rate limited!")
                rate_limited = True
                break

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Completed: {len(progress['completed'])}/{len(BASKET_KEYWORDS)}")

    if progress['completed']:
        print("\nCollected:")
        for name in progress['completed']:
            print(f"  - {name}")

    # If rate limited, show manual instructions for remaining
    still_remaining = [kw for kw in BASKET_KEYWORDS if kw['name'] not in progress['completed']]
    if still_remaining:
        print_manual_instructions(still_remaining)


if __name__ == "__main__":
    main()
