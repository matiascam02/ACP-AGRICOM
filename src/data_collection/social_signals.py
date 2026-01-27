"""
AGRICOM - Social Signals Collection
Collects social media signals related to organic food demand.

Data Sources:
- Instagram hashtag counts (via unofficial API)
- YouTube search trends
- Reddit discussions (via official API)

Usage:
    python social_signals.py

Output:
    data/raw/social_signals_YYYYMMDD.csv
"""

import pandas as pd
import requests
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'raw'

# Hashtags to track
INSTAGRAM_HASHTAGS = [
    'biolebensmittel',
    'biogemüse', 'biogemuese',
    'wochenmarkt',
    'bauernmarkt',
    'bioberlin',
    'farmtotable',
    'sustainablefood',
    'vegandeutschland',
    'healthyeating',
]

# YouTube search terms
YOUTUBE_SEARCHES = [
    'bio kochen',
    'vegane rezepte',
    'nachhaltig leben',
    'wochenmarkt einkauf',
    'meal prep gesund',
]

# Reddit subreddits
REDDIT_SUBREDDITS = [
    'de',  # r/de for German discussions
    'germany',
    'berlin',
    'Kochen',  # German cooking
    'VeganDE',
]


def fetch_youtube_trends(search_terms: list, api_key: str = None) -> pd.DataFrame:
    """
    Fetch YouTube search volume indicators.
    Uses YouTube Data API (requires key) or falls back to Google Trends proxy.
    """
    print("Fetching YouTube trend indicators...")
    
    api_key = api_key or os.environ.get('YOUTUBE_API_KEY')
    
    if not api_key:
        print("  ⚠️  No YouTube API key. Skipping.")
        return pd.DataFrame()
    
    base_url = "https://www.googleapis.com/youtube/v3/search"
    
    results = []
    for term in search_terms:
        try:
            params = {
                'part': 'snippet',
                'q': term,
                'type': 'video',
                'order': 'date',
                'regionCode': 'DE',
                'maxResults': 50,
                'key': api_key
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Count videos from last 30 days
            recent_count = len(data.get('items', []))
            
            results.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'term': term,
                'youtube_recent_videos': recent_count,
                'youtube_total_results': data.get('pageInfo', {}).get('totalResults', 0)
            })
            
            print(f"  ✓ {term}: {recent_count} recent videos")
            time.sleep(0.5)  # Rate limit
            
        except Exception as e:
            print(f"  ✗ {term}: {e}")
    
    return pd.DataFrame(results)


def fetch_reddit_activity(subreddits: list, keywords: list = None) -> pd.DataFrame:
    """
    Fetch Reddit activity using Pushshift or official API.
    Measures post/comment volume related to organic food.
    """
    print("Fetching Reddit activity...")
    
    # Reddit API credentials
    client_id = os.environ.get('REDDIT_CLIENT_ID')
    client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("  ⚠️  No Reddit API credentials. Skipping.")
        print("       Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        return pd.DataFrame()
    
    # Get OAuth token
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {'grant_type': 'client_credentials'}
    headers = {'User-Agent': 'AGRICOM/1.0'}
    
    try:
        token_response = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            auth=auth, data=data, headers=headers, timeout=30
        )
        token_response.raise_for_status()
        token = token_response.json()['access_token']
    except Exception as e:
        print(f"  ✗ OAuth failed: {e}")
        return pd.DataFrame()
    
    headers = {
        'Authorization': f'bearer {token}',
        'User-Agent': 'AGRICOM/1.0'
    }
    
    # Search keywords
    search_keywords = keywords or [
        'bio', 'organic', 'wochenmarkt', 'bauernmarkt', 
        'nachhaltig', 'vegan', 'regional'
    ]
    
    results = []
    for subreddit in subreddits:
        try:
            # Get recent posts
            url = f"https://oauth.reddit.com/r/{subreddit}/search"
            params = {
                'q': ' OR '.join(search_keywords),
                'restrict_sr': True,
                'sort': 'new',
                'limit': 100,
                't': 'month'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get('data', {}).get('children', [])
            
            # Analyze posts
            post_count = len(posts)
            total_score = sum(p['data'].get('score', 0) for p in posts)
            total_comments = sum(p['data'].get('num_comments', 0) for p in posts)
            
            results.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'subreddit': subreddit,
                'organic_posts_30d': post_count,
                'total_score': total_score,
                'total_comments': total_comments,
                'avg_engagement': (total_score + total_comments) / max(post_count, 1)
            })
            
            print(f"  ✓ r/{subreddit}: {post_count} posts, {total_comments} comments")
            time.sleep(1)  # Rate limit
            
        except Exception as e:
            print(f"  ✗ r/{subreddit}: {e}")
    
    return pd.DataFrame(results)


def fetch_instagram_hashtags(hashtags: list) -> pd.DataFrame:
    """
    Fetch Instagram hashtag popularity.
    Note: Requires unofficial API or scraping (use responsibly).
    """
    print("Fetching Instagram hashtag data...")
    
    # Instagram doesn't have a public API for hashtag counts
    # Options:
    # 1. Use RapidAPI's Instagram API (paid)
    # 2. Use browser automation (complex)
    # 3. Manual collection
    
    rapidapi_key = os.environ.get('RAPIDAPI_KEY')
    
    if not rapidapi_key:
        print("  ⚠️  No RapidAPI key. Skipping Instagram.")
        print("       Set RAPIDAPI_KEY for Instagram data")
        return pd.DataFrame()
    
    results = []
    
    for hashtag in hashtags:
        try:
            url = "https://instagram-scraper-api2.p.rapidapi.com/v1/hashtag"
            headers = {
                "X-RapidAPI-Key": rapidapi_key,
                "X-RapidAPI-Host": "instagram-scraper-api2.p.rapidapi.com"
            }
            params = {"hashtag": hashtag}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.ok:
                data = response.json()
                results.append({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'hashtag': hashtag,
                    'post_count': data.get('data', {}).get('count', 0)
                })
                print(f"  ✓ #{hashtag}: {results[-1]['post_count']:,} posts")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"  ✗ #{hashtag}: {e}")
    
    return pd.DataFrame(results)


def create_social_sentiment_index(df_youtube: pd.DataFrame, 
                                   df_reddit: pd.DataFrame,
                                   df_instagram: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite social sentiment index for organic food.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    
    metrics = {'date': today}
    
    # YouTube metrics
    if not df_youtube.empty:
        metrics['youtube_video_count'] = df_youtube['youtube_recent_videos'].sum()
        metrics['youtube_total_reach'] = df_youtube['youtube_total_results'].sum()
    
    # Reddit metrics
    if not df_reddit.empty:
        metrics['reddit_post_count'] = df_reddit['organic_posts_30d'].sum()
        metrics['reddit_engagement'] = df_reddit['avg_engagement'].mean()
        metrics['reddit_comment_volume'] = df_reddit['total_comments'].sum()
    
    # Instagram metrics
    if not df_instagram.empty:
        metrics['instagram_post_count'] = df_instagram['post_count'].sum()
    
    # Calculate composite index (normalized)
    # Higher = more social interest in organic food
    components = []
    if 'youtube_video_count' in metrics:
        components.append(min(metrics['youtube_video_count'] / 50, 1))  # Normalize
    if 'reddit_post_count' in metrics:
        components.append(min(metrics['reddit_post_count'] / 100, 1))
    if 'reddit_engagement' in metrics:
        components.append(min(metrics['reddit_engagement'] / 50, 1))
    
    if components:
        metrics['social_interest_index'] = sum(components) / len(components) * 100
    
    return pd.DataFrame([metrics])


def main():
    print("=" * 60)
    print("AGRICOM - Social Signals Collection")
    print("=" * 60)
    print(f"Target: Organic food social media presence in Germany")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect from various sources
    print("\n1. Collecting YouTube data...")
    df_youtube = fetch_youtube_trends(YOUTUBE_SEARCHES)
    
    print("\n2. Collecting Reddit data...")
    df_reddit = fetch_reddit_activity(REDDIT_SUBREDDITS)
    
    print("\n3. Collecting Instagram data...")
    df_instagram = fetch_instagram_hashtags(INSTAGRAM_HASHTAGS)
    
    # Create composite index
    print("\n4. Creating social sentiment index...")
    df_index = create_social_sentiment_index(df_youtube, df_reddit, df_instagram)
    
    # Save all data
    print("\n5. Saving data...")
    timestamp = datetime.now().strftime('%Y%m%d')
    
    if not df_youtube.empty:
        df_youtube.to_csv(OUTPUT_DIR / f'youtube_signals_{timestamp}.csv', index=False)
    if not df_reddit.empty:
        df_reddit.to_csv(OUTPUT_DIR / f'reddit_signals_{timestamp}.csv', index=False)
    if not df_instagram.empty:
        df_instagram.to_csv(OUTPUT_DIR / f'instagram_signals_{timestamp}.csv', index=False)
    
    # Save composite
    df_index.to_csv(OUTPUT_DIR / f'social_signals_{timestamp}.csv', index=False)
    print(f"   Saved to: social_signals_{timestamp}.csv")
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nData collected:")
    print(f"  YouTube: {'✓' if not df_youtube.empty else '✗'} {len(df_youtube)} terms")
    print(f"  Reddit: {'✓' if not df_reddit.empty else '✗'} {len(df_reddit)} subreddits")
    print(f"  Instagram: {'✓' if not df_instagram.empty else '✗'} {len(df_instagram)} hashtags")
    
    if 'social_interest_index' in df_index.columns:
        print(f"\n📊 Social Interest Index: {df_index['social_interest_index'].iloc[0]:.1f}/100")


if __name__ == "__main__":
    main()
