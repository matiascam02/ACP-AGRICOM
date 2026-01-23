"""
AGRICOM - Reddit Data Collection
Collects posts and comments about organic food, farmers markets, sustainability from German subreddits.

Usage:
    python reddit_collector.py

Note: Requires Reddit API credentials. Get them at https://www.reddit.com/prefs/apps
      Create a "script" type application.
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')

# Subreddits to search (German/Berlin focused)
SUBREDDITS = [
    'berlin',
    'germany',
    'de',
    'kochen',  # German cooking
    'veganDE',
    'Finanzen',  # Sometimes discusses food costs
]

# Search queries (German and English)
SEARCH_QUERIES = [
    'bio lebensmittel',
    'organic food',
    'wochenmarkt',
    'bauernmarkt',
    'farmers market',
    'nachhaltig essen',
    'vegan berlin',
    'supermarkt',
    'REWE bio',
    'Alnatura',
    'Bio Company',
]


def setup_reddit():
    """
    Set up Reddit API connection.

    You need to create credentials at: https://www.reddit.com/prefs/apps
    1. Click "create another app"
    2. Select "script"
    3. Fill in name and redirect URI (http://localhost:8080)
    4. Copy client_id and client_secret
    """
    # Check for environment variables first
    client_id = os.environ.get('REDDIT_CLIENT_ID')
    client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    user_agent = os.environ.get('REDDIT_USER_AGENT', 'AGRICOM Research Bot 1.0')

    if not client_id or not client_secret:
        print("=" * 60)
        print("REDDIT API CREDENTIALS NEEDED")
        print("=" * 60)
        print("\nTo collect Reddit data, you need API credentials:")
        print("\n1. Go to: https://www.reddit.com/prefs/apps")
        print("2. Click 'create another app...' at the bottom")
        print("3. Select 'script' as the app type")
        print("4. Name: 'AGRICOM Research'")
        print("5. Redirect URI: http://localhost:8080")
        print("6. Click 'create app'")
        print("\nThen set environment variables:")
        print("  export REDDIT_CLIENT_ID='your_client_id'")
        print("  export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("\nOr enter them now:")

        client_id = input("Client ID: ").strip()
        client_secret = input("Client Secret: ").strip()

        if not client_id or not client_secret:
            raise ValueError("Reddit credentials required")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    # Verify connection (read-only is fine)
    print(f"Connected to Reddit (read-only: {reddit.read_only})")
    return reddit


def search_subreddit(reddit, subreddit_name, query, limit=100):
    """Search a subreddit for posts matching query."""
    posts = []

    try:
        subreddit = reddit.subreddit(subreddit_name)

        # Search posts
        for post in subreddit.search(query, limit=limit, time_filter='year'):
            posts.append({
                'subreddit': subreddit_name,
                'query': query,
                'post_id': post.id,
                'title': post.title,
                'selftext': post.selftext[:500] if post.selftext else '',
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'url': post.url,
                'author': str(post.author) if post.author else '[deleted]',
            })

    except Exception as e:
        print(f"    Error searching r/{subreddit_name}: {e}")

    return posts


def get_subreddit_top_posts(reddit, subreddit_name, limit=50):
    """Get top posts from a subreddit (food-related)."""
    posts = []

    try:
        subreddit = reddit.subreddit(subreddit_name)

        for post in subreddit.top(time_filter='year', limit=limit):
            # Filter for food-related content
            title_lower = post.title.lower()
            food_keywords = ['essen', 'food', 'kochen', 'cook', 'restaurant',
                           'supermarkt', 'bio', 'organic', 'vegan', 'markt']

            if any(kw in title_lower for kw in food_keywords):
                posts.append({
                    'subreddit': subreddit_name,
                    'query': 'top_food_posts',
                    'post_id': post.id,
                    'title': post.title,
                    'selftext': post.selftext[:500] if post.selftext else '',
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'url': post.url,
                    'author': str(post.author) if post.author else '[deleted]',
                })

    except Exception as e:
        print(f"    Error getting top posts from r/{subreddit_name}: {e}")

    return posts


def main():
    print("=" * 60)
    print("AGRICOM - Reddit Data Collection")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup Reddit connection
    try:
        reddit = setup_reddit()
    except Exception as e:
        print(f"\nFailed to connect to Reddit: {e}")
        print("\nAlternative: Use the Pushshift API or manual collection")
        return

    all_posts = []

    # Search each subreddit with each query
    print(f"\nSearching {len(SUBREDDITS)} subreddits with {len(SEARCH_QUERIES)} queries...")

    for subreddit in SUBREDDITS:
        print(f"\n[r/{subreddit}]")

        # Get top food-related posts
        print(f"  Getting top food posts...")
        top_posts = get_subreddit_top_posts(reddit, subreddit)
        all_posts.extend(top_posts)
        print(f"    Found {len(top_posts)} food-related top posts")

        # Search with each query
        for query in SEARCH_QUERIES:
            print(f"  Searching: '{query}'...")
            posts = search_subreddit(reddit, subreddit, query, limit=50)
            all_posts.extend(posts)
            print(f"    Found {len(posts)} posts")

            # Small delay to be nice to Reddit API
            time.sleep(1)

    # Remove duplicates
    if all_posts:
        df = pd.DataFrame(all_posts)
        df = df.drop_duplicates(subset=['post_id'])

        # Sort by score
        df = df.sort_values('score', ascending=False)

        # Save
        timestamp = datetime.now().strftime('%Y%m%d')
        filepath = os.path.join(OUTPUT_DIR, f'reddit_posts_{timestamp}.csv')
        df.to_csv(filepath, index=False)

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Total unique posts: {len(df)}")
        print(f"Saved to: {filepath}")

        # Summary by subreddit
        print("\nPosts by subreddit:")
        for sub, count in df['subreddit'].value_counts().items():
            print(f"  r/{sub}: {count}")

        # Top posts preview
        print("\nTop 5 posts by score:")
        for _, row in df.head(5).iterrows():
            print(f"  [{row['score']:4d}] r/{row['subreddit']}: {row['title'][:50]}...")
    else:
        print("\nNo posts collected.")


if __name__ == "__main__":
    main()
