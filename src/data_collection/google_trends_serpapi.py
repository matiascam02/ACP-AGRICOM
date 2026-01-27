"""
AGRICOM - Google Trends via SerpAPI
Bypasses rate limits using SerpAPI (requires API key).
Falls back to manual CSV import if no API key available.

Usage:
    # With API key:
    SERPAPI_KEY=your_key python google_trends_serpapi.py
    
    # Without API key (import mode):
    python google_trends_serpapi.py --import-only

Output:
    data/raw/google_trends/trends_*.csv
"""

import pandas as pd
import requests
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'google_trends'
COMBINED_FILE = OUTPUT_DIR / 'all_trends_combined.csv'

# Keywords configuration
KEYWORDS_CONFIG = [
    # Core organic food terms
    {'keyword': 'bio lebensmittel', 'geo': 'DE', 'category': 'organic_food'},
    {'keyword': 'bio gemüse', 'geo': 'DE', 'category': 'organic_food'},
    {'keyword': 'bio obst', 'geo': 'DE', 'category': 'organic_food'},
    {'keyword': 'bio tomaten', 'geo': 'DE', 'category': 'organic_food'},
    {'keyword': 'organic food', 'geo': 'DE', 'category': 'organic_food'},
    
    # Shopping behavior
    {'keyword': 'wochenmarkt berlin', 'geo': 'DE', 'category': 'shopping'},
    {'keyword': 'bauernmarkt', 'geo': 'DE', 'category': 'shopping'},
    {'keyword': 'bio supermarkt', 'geo': 'DE', 'category': 'shopping'},
    
    # Retailers (demand proxies)
    {'keyword': 'alnatura', 'geo': 'DE', 'category': 'retailers'},
    {'keyword': 'bio company', 'geo': 'DE', 'category': 'retailers'},
    {'keyword': 'rewe bio', 'geo': 'DE', 'category': 'retailers'},
    {'keyword': 'edeka bio', 'geo': 'DE', 'category': 'retailers'},
    {'keyword': 'lidl bio', 'geo': 'DE', 'category': 'retailers'},
    
    # Seasonal/recipe signals
    {'keyword': 'grillen rezepte', 'geo': 'DE', 'category': 'seasonal'},
    {'keyword': 'salat rezepte', 'geo': 'DE', 'category': 'seasonal'},
    {'keyword': 'suppe rezepte', 'geo': 'DE', 'category': 'seasonal'},
    {'keyword': 'smoothie rezepte', 'geo': 'DE', 'category': 'seasonal'},
    
    # Lifestyle/sustainability
    {'keyword': 'nachhaltig einkaufen', 'geo': 'DE', 'category': 'sustainability'},
    {'keyword': 'vegan lebensmittel', 'geo': 'DE', 'category': 'sustainability'},
    {'keyword': 'regional einkaufen', 'geo': 'DE', 'category': 'sustainability'},
    {'keyword': 'zero waste', 'geo': 'DE', 'category': 'sustainability'},
]


def fetch_via_serpapi(keyword: str, geo: str = 'DE', api_key: str = None) -> pd.DataFrame:
    """
    Fetch Google Trends data via SerpAPI.
    
    Args:
        keyword: Search term
        geo: Country code (default: DE for Germany)
        api_key: SerpAPI key
        
    Returns:
        DataFrame with date and interest columns
    """
    if not api_key:
        raise ValueError("SerpAPI key required")
    
    url = "https://serpapi.com/search.json"
    params = {
        'engine': 'google_trends',
        'q': keyword,
        'geo': geo,
        'data_type': 'TIMESERIES',
        'date': 'today 5-y',  # Last 5 years
        'api_key': api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()
    
    if 'interest_over_time' not in data:
        return pd.DataFrame()
    
    timeline = data['interest_over_time']['timeline_data']
    
    rows = []
    for point in timeline:
        rows.append({
            'date': point['date'],
            'value': point['values'][0]['extracted_value']
        })
    
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['keyword'] = keyword
    
    return df


def import_manual_csvs() -> pd.DataFrame:
    """
    Import manually downloaded Google Trends CSVs.
    Handles various CSV formats from Google Trends export.
    
    Returns:
        Combined DataFrame with all trends data
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    for csv_file in OUTPUT_DIR.glob('*.csv'):
        if csv_file.name in ['all_trends_combined.csv', 'progress.json']:
            continue
            
        try:
            # Try standard format first
            df = pd.read_csv(csv_file, skiprows=1)  # Skip category header
            
            if len(df.columns) >= 2:
                # Rename columns
                date_col = df.columns[0]
                value_col = df.columns[1]
                
                df = df.rename(columns={date_col: 'date', value_col: 'value'})
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                
                # Extract keyword from filename
                keyword = csv_file.stem.replace('trends_', '').replace('_de_5y', '').replace('_', ' ')
                df['keyword'] = keyword
                df['source_file'] = csv_file.name
                
                all_data.append(df[['date', 'value', 'keyword', 'source_file']])
                print(f"  ✓ Imported: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"  ✗ Failed to import {csv_file.name}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def get_collection_status() -> dict:
    """Get status of what's been collected."""
    status = {
        'collected': [],
        'missing': [],
        'total_keywords': len(KEYWORDS_CONFIG)
    }
    
    existing_files = set(f.stem.lower() for f in OUTPUT_DIR.glob('*.csv'))
    
    for kw in KEYWORDS_CONFIG:
        # Generate possible filenames
        clean_keyword = kw['keyword'].replace(' ', '_').replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe')
        possible_names = [
            f"trends_{clean_keyword}_de_5y",
            f"{clean_keyword}_de",
            clean_keyword,
        ]
        
        found = any(name.lower() in existing_files for name in possible_names)
        
        if found:
            status['collected'].append(kw['keyword'])
        else:
            status['missing'].append(kw)
    
    return status


def generate_manual_download_guide(missing_keywords: list) -> str:
    """Generate instructions for manual download."""
    guide = """
# Google Trends Manual Download Guide
# ====================================

## Instructions:
1. Go to https://trends.google.com/trends/
2. Enter keyword in search box
3. Set Location: Germany
4. Set Time range: Past 5 years
5. Click download icon (↓) top-right of chart
6. Save CSV to: data/raw/google_trends/

## Keywords to Download:
"""
    for kw in missing_keywords:
        guide += f"\n### {kw['keyword']}\n"
        guide += f"- Category: {kw['category']}\n"
        guide += f"- Save as: trends_{kw['keyword'].replace(' ', '_')}_de_5y.csv\n"
    
    return guide


def combine_all_trends() -> pd.DataFrame:
    """Combine all trends data into single analysis-ready file."""
    df = import_manual_csvs()
    
    if df.empty:
        print("No trends data found!")
        return df
    
    # Standardize dates to weekly (Sunday)
    df['week'] = df['date'].dt.to_period('W').dt.start_time
    
    # Pivot to wide format for analysis
    df_wide = df.pivot_table(
        index='week',
        columns='keyword',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Save combined file
    df_wide.to_csv(COMBINED_FILE, index=False)
    print(f"\nSaved combined trends to: {COMBINED_FILE}")
    
    return df_wide


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Google Trends data collection')
    parser.add_argument('--import-only', action='store_true', help='Only import existing CSVs')
    parser.add_argument('--status', action='store_true', help='Show collection status')
    parser.add_argument('--guide', action='store_true', help='Generate manual download guide')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AGRICOM - Google Trends Collection")
    print("=" * 60)
    
    # Get status
    status = get_collection_status()
    print(f"\nCollection Status:")
    print(f"  Collected: {len(status['collected'])}/{status['total_keywords']}")
    print(f"  Missing: {len(status['missing'])}")
    
    if args.status:
        print("\nCollected keywords:")
        for kw in status['collected']:
            print(f"  ✓ {kw}")
        print("\nMissing keywords:")
        for kw in status['missing']:
            print(f"  ✗ {kw['keyword']} ({kw['category']})")
        return
    
    if args.guide:
        guide = generate_manual_download_guide(status['missing'])
        guide_path = OUTPUT_DIR / 'DOWNLOAD_GUIDE.md'
        with open(guide_path, 'w') as f:
            f.write(guide)
        print(f"\nGenerated download guide: {guide_path}")
        print(guide)
        return
    
    # Try SerpAPI first
    api_key = os.environ.get('SERPAPI_KEY') or os.environ.get('SERPAPI_API_KEY')
    
    if api_key and not args.import_only:
        print("\n📡 Collecting via SerpAPI...")
        for kw in status['missing'][:5]:  # Limit to 5 per run (API costs)
            try:
                df = fetch_via_serpapi(kw['keyword'], kw['geo'], api_key)
                if not df.empty:
                    filename = f"trends_{kw['keyword'].replace(' ', '_')}_de_5y.csv"
                    df.to_csv(OUTPUT_DIR / filename, index=False)
                    print(f"  ✓ Collected: {kw['keyword']}")
                time.sleep(2)  # Rate limit courtesy
            except Exception as e:
                print(f"  ✗ Failed: {kw['keyword']} - {e}")
    else:
        print("\n⚠️  No SerpAPI key found. Using import mode.")
        print("   Set SERPAPI_KEY environment variable to enable API collection.")
    
    # Import and combine all data
    print("\n📁 Importing existing CSV files...")
    df_combined = combine_all_trends()
    
    if not df_combined.empty:
        print(f"\n✅ Combined dataset: {len(df_combined)} weeks, {len(df_combined.columns)-1} keywords")
    
    # Show what's still missing
    if status['missing']:
        print(f"\n⚠️  Still missing {len(status['missing'])} keywords.")
        print("   Run with --guide to generate manual download instructions.")


if __name__ == "__main__":
    main()
