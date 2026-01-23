"""
AGRICOM - GDELT News Sentiment Analysis
Analyzes news coverage and sentiment about organic food topics in Germany.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
OUTPUT_FIGURES = os.path.join(os.path.dirname(__file__), '../../outputs/figures')
OUTPUT_REPORTS = os.path.join(os.path.dirname(__file__), '../../outputs/reports')

os.makedirs(OUTPUT_FIGURES, exist_ok=True)
os.makedirs(OUTPUT_REPORTS, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'organic_food': '#2ecc71',
    'organic_produce': '#27ae60',
    'food_safety': '#e74c3c',
    'sustainable_food': '#3498db',
    'farmers_markets': '#f39c12',
    'plant_based': '#9b59b6',
    'primary': '#2ecc71',
    'secondary': '#3498db'
}


def load_gdelt_articles():
    """Load GDELT articles data."""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('gdelt_articles_') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No GDELT articles found")

    filepath = os.path.join(DATA_DIR, sorted(files)[-1])
    print(f"Loading articles: {filepath}")

    df = pd.read_csv(filepath)
    df['seendate'] = pd.to_datetime(df['seendate'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    return df


def load_gdelt_timeline():
    """Load GDELT timeline/tone data."""
    files = [f for f in os.listdir(DATA_DIR) if f.startswith('gdelt_timeline_') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No GDELT timeline found")

    filepath = os.path.join(DATA_DIR, sorted(files)[-1])
    print(f"Loading timeline: {filepath}")

    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def plot_article_coverage(articles_df, save=True):
    """Plot article counts by topic and over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Article counts by topic
    ax1 = axes[0]
    topic_counts = articles_df['query_name'].value_counts()
    colors_list = [COLORS.get(topic, '#95a5a6') for topic in topic_counts.index]

    bars = ax1.barh(topic_counts.index, topic_counts.values, color=colors_list, alpha=0.8)
    ax1.set_xlabel('Number of Articles')
    ax1.set_title('GDELT News Coverage by Topic (Germany, Last 2 Years)',
                  fontsize=14, fontweight='bold')

    # Add count labels
    for bar, count in zip(bars, topic_counts.values):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 str(count), va='center', fontsize=10)

    # Plot 2: Articles over time
    ax2 = axes[1]
    articles_df['month'] = articles_df['seendate'].dt.to_period('M')
    monthly_counts = articles_df.groupby(['month', 'query_name']).size().unstack(fill_value=0)

    # Convert period index to timestamp for plotting
    monthly_counts.index = monthly_counts.index.to_timestamp()

    for topic in monthly_counts.columns:
        color = COLORS.get(topic, '#95a5a6')
        ax2.plot(monthly_counts.index, monthly_counts[topic],
                 label=topic.replace('_', ' ').title(),
                 color=color, linewidth=2, marker='o', markersize=4)

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Articles')
    ax2.set_title('Monthly Article Volume by Topic', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'gdelt_article_coverage.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_sentiment_analysis(timeline_df, save=True):
    """Plot sentiment/tone analysis over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Sentiment over time by topic
    ax1 = axes[0]

    for topic in timeline_df['query_name'].unique():
        topic_data = timeline_df[timeline_df['query_name'] == topic].copy()
        topic_data = topic_data.set_index('datetime').sort_index()

        # Weekly rolling average for smoothing
        topic_data['tone_smooth'] = topic_data['Average Tone'].rolling(window=7, min_periods=1).mean()

        color = COLORS.get(topic, '#95a5a6')
        ax1.plot(topic_data.index, topic_data['tone_smooth'],
                 label=topic.replace('_', ' ').title(),
                 color=color, linewidth=1.5, alpha=0.8)

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Sentiment Score (Tone)')
    ax1.set_title('News Sentiment Over Time (7-day Rolling Average)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Add annotations for positive/negative zones
    ax1.text(0.02, 0.95, 'Positive Coverage ↑', transform=ax1.transAxes,
             fontsize=9, color='green', alpha=0.7)
    ax1.text(0.02, 0.05, 'Negative Coverage ↓', transform=ax1.transAxes,
             fontsize=9, color='red', alpha=0.7)

    # Plot 2: Average sentiment by topic
    ax2 = axes[1]
    avg_sentiment = timeline_df.groupby('query_name')['Average Tone'].mean().sort_values(ascending=True)

    colors_list = []
    for topic in avg_sentiment.index:
        val = avg_sentiment[topic]
        if val > 0:
            colors_list.append('#2ecc71')  # Green for positive
        else:
            colors_list.append('#e74c3c')  # Red for negative

    bars = ax2.barh(avg_sentiment.index, avg_sentiment.values, color=colors_list, alpha=0.8)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Average Sentiment Score')
    ax2.set_title('Average News Sentiment by Topic', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, avg_sentiment.values):
        ax2.text(val + 0.05 if val > 0 else val - 0.3,
                 bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=10)

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'gdelt_sentiment_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_source_analysis(articles_df, save=True):
    """Plot analysis of news sources."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Top sources
    ax1 = axes[0]
    source_counts = articles_df['domain'].value_counts().head(15)

    ax1.barh(source_counts.index, source_counts.values,
             color=COLORS['secondary'], alpha=0.8)
    ax1.set_xlabel('Number of Articles')
    ax1.set_title('Top 15 News Sources', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    # Plot 2: Language distribution
    ax2 = axes[1]
    lang_counts = articles_df['language'].value_counts()
    colors_lang = ['#2ecc71' if lang == 'German' else '#3498db' for lang in lang_counts.index]

    ax2.pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%',
            colors=colors_lang, startangle=90, explode=[0.05] * len(lang_counts))
    ax2.set_title('Article Language Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'gdelt_source_analysis.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def generate_report(articles_df, timeline_df):
    """Generate summary report."""
    report = []
    report.append("=" * 60)
    report.append("AGRICOM - GDELT News Analysis Report")
    report.append("=" * 60)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Article summary
    report.append("\n" + "-" * 40)
    report.append("ARTICLE SUMMARY")
    report.append("-" * 40)
    report.append(f"Total articles collected: {len(articles_df)}")
    report.append(f"Date range: {articles_df['seendate'].min().strftime('%Y-%m-%d')} to {articles_df['seendate'].max().strftime('%Y-%m-%d')}")
    report.append(f"Unique sources: {articles_df['domain'].nunique()}")

    report.append("\nArticles by topic:")
    for topic, count in articles_df['query_name'].value_counts().items():
        report.append(f"  - {topic.replace('_', ' ').title()}: {count}")

    # Sentiment summary
    report.append("\n" + "-" * 40)
    report.append("SENTIMENT SUMMARY")
    report.append("-" * 40)

    for topic in timeline_df['query_name'].unique():
        topic_data = timeline_df[timeline_df['query_name'] == topic]
        avg_tone = topic_data['Average Tone'].mean()
        sentiment = "Positive" if avg_tone > 0 else "Negative"
        report.append(f"  - {topic.replace('_', ' ').title()}: {avg_tone:.2f} ({sentiment})")

    # Top sources
    report.append("\n" + "-" * 40)
    report.append("TOP NEWS SOURCES")
    report.append("-" * 40)
    for source, count in articles_df['domain'].value_counts().head(10).items():
        report.append(f"  - {source}: {count} articles")

    # Key headlines
    report.append("\n" + "-" * 40)
    report.append("SAMPLE HEADLINES")
    report.append("-" * 40)
    for _, row in articles_df.head(10).iterrows():
        title = str(row.get('title', 'No title'))[:70]
        date = row['seendate'].strftime('%Y-%m-%d') if pd.notna(row['seendate']) else 'N/A'
        report.append(f"  [{date}] {title}...")

    report.append("\n" + "=" * 60)

    # Save report
    report_text = "\n".join(report)
    print(report_text)

    report_path = os.path.join(OUTPUT_REPORTS, 'gdelt_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nSaved report to: {report_path}")

    return report_text


def main():
    print("=" * 60)
    print("AGRICOM - GDELT News Sentiment Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    articles_df = load_gdelt_articles()
    timeline_df = load_gdelt_timeline()

    print(f"   Articles: {len(articles_df)}")
    print(f"   Timeline points: {len(timeline_df)}")

    # Generate visualizations
    print("\n2. Generating visualizations...")

    print("   - Article coverage...")
    plot_article_coverage(articles_df)

    print("   - Sentiment analysis...")
    plot_sentiment_analysis(timeline_df)

    print("   - Source analysis...")
    plot_source_analysis(articles_df)

    # Generate report
    print("\n3. Generating report...")
    generate_report(articles_df, timeline_df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
