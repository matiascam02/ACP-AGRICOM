"""
AGRICOM - GDELT News Data Collection
Collects news articles and sentiment about organic food, sustainability, food safety in Germany.

Usage:
    python gdelt_news.py

Output:
    data/raw/gdelt_news_YYYYMMDD.csv
    data/raw/gdelt_timeline_YYYYMMDD.csv
"""

import pandas as pd
from gdeltdoc import GdeltDoc, Filters
from datetime import datetime, timedelta
import os
import time

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')

# Search queries - German and English terms
QUERIES = [
    # Organic food
    {
        'name': 'organic_food',
        'keywords': ['bio lebensmittel', 'organic food', 'ökologisch'],
        'description': 'Organic food coverage'
    },
    {
        'name': 'organic_produce',
        'keywords': ['bio gemüse', 'bio obst', 'organic vegetables'],
        'description': 'Organic produce coverage'
    },
    # Food safety / scandals
    {
        'name': 'food_safety',
        'keywords': ['lebensmittelskandal', 'pestizide lebensmittel', 'food safety germany'],
        'description': 'Food safety and scandals'
    },
    # Sustainability
    {
        'name': 'sustainable_food',
        'keywords': ['nachhaltige ernährung', 'sustainable food', 'klimafreundlich essen'],
        'description': 'Sustainable food coverage'
    },
    # Farmers markets
    {
        'name': 'farmers_markets',
        'keywords': ['wochenmarkt', 'bauernmarkt', 'farmers market germany'],
        'description': 'Farmers market coverage'
    },
    # Vegan/vegetarian trends
    {
        'name': 'plant_based',
        'keywords': ['vegan deutschland', 'vegetarisch trend', 'plant based germany'],
        'description': 'Plant-based food trends'
    },
]


def search_articles(gd, keywords, start_date, end_date, max_records=250):
    """
    Search GDELT for articles matching keywords.

    Args:
        gd: GdeltDoc instance
        keywords: List of search terms
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        max_records: Maximum articles to return

    Returns:
        DataFrame with articles
    """
    try:
        f = Filters(
            keyword=keywords,
            start_date=start_date,
            end_date=end_date,
            country='GM'  # Germany
        )

        articles = gd.article_search(f)

        if articles is not None and len(articles) > 0:
            # Limit records
            if len(articles) > max_records:
                articles = articles.head(max_records)
            return articles
        return pd.DataFrame()

    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def get_timeline_tone(gd, keywords, start_date, end_date):
    """
    Get timeline of article tone/sentiment for keywords.

    Args:
        gd: GdeltDoc instance
        keywords: List of search terms
        start_date: Start date string
        end_date: End date string

    Returns:
        DataFrame with daily tone values
    """
    try:
        f = Filters(
            keyword=keywords,
            start_date=start_date,
            end_date=end_date,
            country='GM'
        )

        timeline = gd.timeline_search("timelinetone", f)
        return timeline

    except Exception as e:
        print(f"    Timeline error: {e}")
        return pd.DataFrame()


def get_article_counts(gd, keywords, start_date, end_date):
    """
    Get timeline of article counts for keywords.
    """
    try:
        f = Filters(
            keyword=keywords,
            start_date=start_date,
            end_date=end_date,
            country='GM'
        )

        timeline = gd.timeline_search("timelinevol", f)
        return timeline

    except Exception as e:
        print(f"    Count error: {e}")
        return pd.DataFrame()


def main():
    print("=" * 60)
    print("AGRICOM - GDELT News Data Collection")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize GDELT
    gd = GdeltDoc()

    # Date range - last 2 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    print(f"Date range: {start_date} to {end_date}")
    print(f"Country: Germany (GM)")
    print(f"Queries: {len(QUERIES)}")
    print("=" * 60)

    all_articles = []
    all_timelines = []

    for i, query in enumerate(QUERIES):
        print(f"\n[{i+1}/{len(QUERIES)}] {query['name']}: {query['description']}")
        print(f"    Keywords: {query['keywords']}")

        # Get articles
        print("    Fetching articles...")
        articles = search_articles(gd, query['keywords'], start_date, end_date)

        if not articles.empty:
            articles['query_name'] = query['name']
            all_articles.append(articles)
            print(f"    Found {len(articles)} articles")
        else:
            print("    No articles found")

        # Get tone timeline
        print("    Fetching tone timeline...")
        timeline = get_timeline_tone(gd, query['keywords'], start_date, end_date)

        if timeline is not None and not timeline.empty:
            timeline['query_name'] = query['name']
            all_timelines.append(timeline)
            print(f"    Got {len(timeline)} timeline points")

        # Small delay between queries
        time.sleep(2)

    # Combine and save articles
    if all_articles:
        combined_articles = pd.concat(all_articles, ignore_index=True)
        timestamp = datetime.now().strftime('%Y%m%d')

        articles_path = os.path.join(OUTPUT_DIR, f'gdelt_articles_{timestamp}.csv')
        combined_articles.to_csv(articles_path, index=False)
        print(f"\nSaved {len(combined_articles)} articles to: {articles_path}")

        # Print sample
        print("\nSample articles:")
        if 'title' in combined_articles.columns:
            for _, row in combined_articles.head(5).iterrows():
                title = str(row.get('title', 'No title'))[:60]
                tone = row.get('tone', 0)
                try:
                    print(f"  [{float(tone):.1f}] {title}...")
                except:
                    print(f"  [N/A] {title}...")

    # Combine and save timelines
    if all_timelines:
        combined_timeline = pd.concat(all_timelines, ignore_index=True)

        timeline_path = os.path.join(OUTPUT_DIR, f'gdelt_timeline_{timestamp}.csv')
        combined_timeline.to_csv(timeline_path, index=False)
        print(f"\nSaved timeline to: {timeline_path}")

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
