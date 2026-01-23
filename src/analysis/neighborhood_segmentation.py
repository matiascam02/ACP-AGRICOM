"""
AGRICOM - Berlin Neighborhood Segmentation Analysis
Profiles key Berlin neighborhoods for organic produce demand targeting.

Target Neighborhoods:
1. Mitte - Wealthy professionals, high disposable income
2. Charlottenburg - Artistic/creative, quality-focused
3. Kreuzberg - Eco-conscious, alternative lifestyle

Data sources: Berlin statistical data, demographic profiles, market research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import os

# Configuration
OUTPUT_FIGURES = os.path.join(os.path.dirname(__file__), '../../outputs/figures')
OUTPUT_REPORTS = os.path.join(os.path.dirname(__file__), '../../outputs/reports')

os.makedirs(OUTPUT_FIGURES, exist_ok=True)
os.makedirs(OUTPUT_REPORTS, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

# Berlin Neighborhood Data (based on statistical research and market analysis)
NEIGHBORHOODS = {
    'Mitte': {
        'population': 384172,
        'avg_income': 42000,  # EUR/year
        'age_median': 38,
        'foreign_pct': 32.5,
        'green_voters_pct': 21.3,  # 2021 election
        'organic_stores': 45,  # Estimated bio shops
        'farmers_markets': 8,
        'profile': 'Wealthy professionals',
        'characteristics': [
            'High disposable income',
            'International residents',
            'Premium quality seekers',
            'Convenience-oriented',
            'Restaurant/dining culture'
        ],
        'organic_affinity': 0.75,  # 0-1 scale
        'price_sensitivity': 0.3,  # 0-1, lower = less sensitive
        'convenience_priority': 0.8,
        'color': '#3498db'
    },
    'Charlottenburg-Wilmersdorf': {
        'population': 342332,
        'avg_income': 38500,
        'age_median': 45,
        'foreign_pct': 24.8,
        'green_voters_pct': 19.8,
        'organic_stores': 38,
        'farmers_markets': 6,
        'profile': 'Established families & creatives',
        'characteristics': [
            'Older demographic',
            'Cultural appreciation',
            'Quality over quantity',
            'Brand loyal',
            'Traditional shopping habits'
        ],
        'organic_affinity': 0.65,
        'price_sensitivity': 0.4,
        'convenience_priority': 0.5,
        'color': '#9b59b6'
    },
    'Friedrichshain-Kreuzberg': {
        'population': 289201,
        'avg_income': 28000,
        'age_median': 34,
        'foreign_pct': 29.1,
        'green_voters_pct': 32.4,  # Highest Green vote in Berlin
        'organic_stores': 52,
        'farmers_markets': 12,
        'profile': 'Eco-conscious & alternative',
        'characteristics': [
            'Younger demographic',
            'Highest environmental awareness',
            'Price-conscious but values-driven',
            'Local/community focused',
            'Anti-corporate sentiment'
        ],
        'organic_affinity': 0.85,
        'price_sensitivity': 0.7,
        'convenience_priority': 0.4,
        'color': '#2ecc71'
    }
}

# Comparison neighborhoods (for context)
COMPARISON = {
    'Neukölln': {
        'population': 329917,
        'avg_income': 24000,
        'organic_affinity': 0.45,
        'color': '#e74c3c'
    },
    'Pankow': {
        'population': 410716,
        'avg_income': 32000,
        'organic_affinity': 0.60,
        'color': '#f39c12'
    }
}


def create_segment_profiles():
    """Create detailed segment profiles DataFrame."""
    data = []
    for name, info in NEIGHBORHOODS.items():
        data.append({
            'Neighborhood': name,
            'Population': info['population'],
            'Avg Income (EUR)': info['avg_income'],
            'Median Age': info['age_median'],
            'Foreign %': info['foreign_pct'],
            'Green Vote %': info['green_voters_pct'],
            'Organic Stores': info['organic_stores'],
            'Farmers Markets': info['farmers_markets'],
            'Organic Affinity': info['organic_affinity'],
            'Price Sensitivity': info['price_sensitivity'],
            'Profile': info['profile']
        })
    return pd.DataFrame(data)


def plot_neighborhood_comparison(save=True):
    """Create comparison visualization of neighborhoods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    neighborhoods = list(NEIGHBORHOODS.keys())
    colors = [NEIGHBORHOODS[n]['color'] for n in neighborhoods]

    # Plot 1: Income vs Organic Affinity
    ax1 = axes[0, 0]
    incomes = [NEIGHBORHOODS[n]['avg_income'] for n in neighborhoods]
    affinities = [NEIGHBORHOODS[n]['organic_affinity'] for n in neighborhoods]
    populations = [NEIGHBORHOODS[n]['population'] / 5000 for n in neighborhoods]  # Scale for bubble size

    scatter = ax1.scatter(incomes, affinities, s=populations, c=colors, alpha=0.7, edgecolor='white', linewidth=2)

    for i, name in enumerate(neighborhoods):
        short_name = name.split('-')[0]  # Shorten for display
        ax1.annotate(short_name, (incomes[i], affinities[i]),
                    xytext=(10, 5), textcoords='offset points', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Average Income (EUR/year)', fontsize=11)
    ax1.set_ylabel('Organic Affinity Score', fontsize=11)
    ax1.set_title('Income vs Organic Affinity\n(bubble size = population)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.5, 1.0)

    # Plot 2: Green Vote % vs Organic Stores
    ax2 = axes[0, 1]
    green_votes = [NEIGHBORHOODS[n]['green_voters_pct'] for n in neighborhoods]
    organic_stores = [NEIGHBORHOODS[n]['organic_stores'] for n in neighborhoods]

    bars = ax2.bar(neighborhoods, green_votes, color=colors, alpha=0.7, edgecolor='white', linewidth=2)

    # Add store count as text
    for i, (bar, stores) in enumerate(zip(bars, organic_stores)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{stores} stores', ha='center', fontsize=9)

    ax2.set_ylabel('Green Party Vote % (2021)', fontsize=11)
    ax2.set_title('Environmental Consciousness\n(numbers = organic stores)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels([n.split('-')[0] for n in neighborhoods], rotation=15, ha='right')

    # Plot 3: Radar chart - Segment characteristics
    ax3 = axes[1, 0]

    categories = ['Organic\nAffinity', 'Price\nTolerance', 'Convenience\nPriority',
                  'Income\n(normalized)', 'Green\nVote']

    # Normalize values to 0-1 scale
    max_income = max(NEIGHBORHOODS[n]['avg_income'] for n in neighborhoods)
    max_green = max(NEIGHBORHOODS[n]['green_voters_pct'] for n in neighborhoods)

    for name in neighborhoods:
        n = NEIGHBORHOODS[name]
        values = [
            n['organic_affinity'],
            1 - n['price_sensitivity'],  # Invert so higher = more tolerant
            n['convenience_priority'],
            n['avg_income'] / max_income,
            n['green_voters_pct'] / max_green
        ]
        values.append(values[0])  # Close the polygon

        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles.append(angles[0])

        ax3.plot(angles, values, 'o-', linewidth=2, label=name.split('-')[0], color=n['color'])
        ax3.fill(angles, values, alpha=0.15, color=n['color'])

    ax3.set_xticks(np.linspace(0, 2*np.pi, len(categories), endpoint=False))
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax3.set_title('Neighborhood Segment Profiles', fontsize=12, fontweight='bold')

    # Plot 4: Market opportunity matrix
    ax4 = axes[1, 1]

    # Calculate opportunity score: affinity * population * (1/price_sensitivity)
    opportunity_scores = []
    for name in neighborhoods:
        n = NEIGHBORHOODS[name]
        score = n['organic_affinity'] * (n['population'] / 100000) * (1 / (n['price_sensitivity'] + 0.1))
        opportunity_scores.append(score)

    # Horizontal bar chart
    y_pos = range(len(neighborhoods))
    bars = ax4.barh(y_pos, opportunity_scores, color=colors, alpha=0.7, edgecolor='white', linewidth=2)

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([n.split('-')[0] for n in neighborhoods])
    ax4.set_xlabel('Market Opportunity Score', fontsize=11)
    ax4.set_title('Organic Produce Market Opportunity\n(affinity x population / price sensitivity)', fontsize=12, fontweight='bold')

    # Add score labels
    for bar, score in zip(bars, opportunity_scores):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'neighborhood_segmentation.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def plot_demand_drivers(save=True):
    """Create visualization of demand drivers by neighborhood."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    neighborhoods = list(NEIGHBORHOODS.keys())

    # Define demand drivers and their weights per neighborhood
    drivers = {
        'Mitte': {
            'Quality Premium': 0.9,
            'Convenience': 0.85,
            'Health Benefits': 0.7,
            'Environmental': 0.5,
            'Local/Community': 0.3
        },
        'Charlottenburg-Wilmersdorf': {
            'Quality Premium': 0.8,
            'Convenience': 0.5,
            'Health Benefits': 0.75,
            'Environmental': 0.55,
            'Local/Community': 0.6
        },
        'Friedrichshain-Kreuzberg': {
            'Quality Premium': 0.5,
            'Convenience': 0.4,
            'Health Benefits': 0.65,
            'Environmental': 0.95,
            'Local/Community': 0.9
        }
    }

    for i, (name, driver_scores) in enumerate(drivers.items()):
        ax = axes[i]
        labels = list(driver_scores.keys())
        values = list(driver_scores.values())

        # Create horizontal bar chart
        y_pos = range(len(labels))
        color = NEIGHBORHOODS[name]['color']

        bars = ax.barh(y_pos, values, color=color, alpha=0.7, edgecolor='white', linewidth=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Importance Score')
        ax.set_title(f"{name.split('-')[0]}\n{NEIGHBORHOODS[name]['profile']}", fontsize=11, fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.0%}', va='center', fontsize=9)

    plt.suptitle('Purchase Decision Drivers by Neighborhood', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_FIGURES, 'neighborhood_demand_drivers.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    plt.close()


def generate_segmentation_report():
    """Generate detailed segmentation report."""
    report = []
    report.append("=" * 70)
    report.append("AGRICOM - Berlin Neighborhood Segmentation Report")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Executive Summary
    report.append("\n" + "-" * 50)
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 50)
    report.append("""
Three Berlin neighborhoods have been identified as primary targets for
organic produce demand forecasting, each representing distinct consumer
segments with different motivations and purchasing behaviors.
""")

    # Segment Details
    for name, info in NEIGHBORHOODS.items():
        report.append("\n" + "=" * 50)
        report.append(f"SEGMENT: {name.upper()}")
        report.append("=" * 50)
        report.append(f"\nProfile: {info['profile']}")
        report.append(f"Population: {info['population']:,}")
        report.append(f"Average Income: EUR {info['avg_income']:,}/year")
        report.append(f"Median Age: {info['age_median']}")
        report.append(f"Organic Affinity Score: {info['organic_affinity']:.0%}")

        report.append("\nKey Characteristics:")
        for char in info['characteristics']:
            report.append(f"  - {char}")

        report.append("\nMarket Infrastructure:")
        report.append(f"  - Organic stores: {info['organic_stores']}")
        report.append(f"  - Farmers markets: {info['farmers_markets']}")

        report.append(f"\nGreen Party Vote (2021): {info['green_voters_pct']}%")

    # Recommendations
    report.append("\n" + "=" * 50)
    report.append("TARGETING RECOMMENDATIONS")
    report.append("=" * 50)

    report.append("""
1. FRIEDRICHSHAIN-KREUZBERG (Primary Target)
   - Highest organic affinity (85%)
   - Strongest environmental values
   - Most farmers markets (12)
   - Focus: Local sourcing, sustainability messaging
   - Channel: Farmers markets, Bio Company, community events

2. MITTE (Secondary Target)
   - High income, premium positioning
   - Convenience-driven purchases
   - Focus: Quality, health benefits, convenience
   - Channel: REWE Bio, delivery services, upscale organic shops

3. CHARLOTTENBURG-WILMERSDORF (Tertiary Target)
   - Quality-focused, brand loyal
   - Traditional shopping preferences
   - Focus: Product quality, trusted brands
   - Channel: Established organic retailers, weekly markets
""")

    # Data signals by segment
    report.append("\n" + "=" * 50)
    report.append("RECOMMENDED DATA SIGNALS BY SEGMENT")
    report.append("=" * 50)

    report.append("""
KREUZBERG:
  - Google Trends: "bio", "nachhaltig", "wochenmarkt kreuzberg"
  - Weather: Farmers market attendance (warm weekends)
  - Events: Karneval der Kulturen, neighborhood festivals
  - Social: r/berlin eco discussions, Instagram #kreuzbergfood

MITTE:
  - Google Trends: "bio lieferung", "organic delivery berlin"
  - Weather: Less impactful (convenience-oriented)
  - Events: Business conferences, tourist seasons
  - Social: Expat forums, professional networks

CHARLOTTENBURG:
  - Google Trends: "bio qualität", "alnatura", "reformhaus"
  - Weather: Weekend shopping patterns
  - Events: Cultural events, opera/theater seasons
  - Social: Traditional media, local newspapers
""")

    report.append("\n" + "=" * 70)

    # Save report
    report_text = "\n".join(report)
    print(report_text)

    report_path = os.path.join(OUTPUT_REPORTS, 'neighborhood_segmentation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nSaved report to: {report_path}")

    return report_text


def main():
    print("=" * 60)
    print("AGRICOM - Berlin Neighborhood Segmentation")
    print("=" * 60)

    # Create profile DataFrame
    print("\n1. Creating segment profiles...")
    profiles_df = create_segment_profiles()
    print(profiles_df.to_string())

    # Save profiles
    profiles_path = os.path.join(OUTPUT_REPORTS, 'neighborhood_profiles.csv')
    profiles_df.to_csv(profiles_path, index=False)
    print(f"\nSaved profiles to: {profiles_path}")

    # Generate visualizations
    print("\n2. Generating visualizations...")

    print("   - Neighborhood comparison...")
    plot_neighborhood_comparison()

    print("   - Demand drivers...")
    plot_demand_drivers()

    # Generate report
    print("\n3. Generating report...")
    generate_segmentation_report()

    print("\n" + "=" * 60)
    print("SEGMENTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
