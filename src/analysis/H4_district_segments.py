"""
AGRICOM - H4: District Segment Intensity
Tests: "Do districts with higher 'Premium Sustainability Buyers' share show
        disproportionately higher organic search intensity?"

Method: OLS with district-level pseudo-panel using hardcoded neighborhood data.
Uses city-wide GT scaled by district organic affinity as proxy.

Input:  data/processed/master_panel_*.csv + neighborhood_segmentation.py data
Output: outputs/tables/H4_district_coefficients.csv
        outputs/figures/H4_berlin_choropleth.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
FIGURE_DIR = PROJECT_DIR / 'outputs' / 'figures'
TABLE_DIR = PROJECT_DIR / 'outputs' / 'tables'

# Hardcoded neighborhood data (from neighborhood_segmentation.py)
NEIGHBORHOODS = {
    'Mitte': {
        'population': 384172, 'avg_income': 42000, 'green_voters_pct': 21.3,
        'organic_stores': 45, 'organic_affinity': 0.75, 'price_sensitivity': 0.3,
        'primary_segment': 'Convenience-Driven Urban',
    },
    'Charlottenburg-Wilmersdorf': {
        'population': 342332, 'avg_income': 38500, 'green_voters_pct': 19.8,
        'organic_stores': 38, 'organic_affinity': 0.65, 'price_sensitivity': 0.4,
        'primary_segment': 'Community-Focused Traditionalists',
    },
    'Friedrichshain-Kreuzberg': {
        'population': 289201, 'avg_income': 28000, 'green_voters_pct': 32.4,
        'organic_stores': 52, 'organic_affinity': 0.85, 'price_sensitivity': 0.7,
        'primary_segment': 'Premium Sustainability Buyers',
    },
    'Neukoelln': {
        'population': 329917, 'avg_income': 24000, 'green_voters_pct': 16.5,
        'organic_stores': 18, 'organic_affinity': 0.35, 'price_sensitivity': 0.8,
        'primary_segment': 'Price-Sensitive Discounters',
    },
    'Pankow': {
        'population': 407765, 'avg_income': 33000, 'green_voters_pct': 22.1,
        'organic_stores': 30, 'organic_affinity': 0.55, 'price_sensitivity': 0.5,
        'primary_segment': 'Modern Plant-Based Urbanites',
    },
}


def load_panel():
    files = sorted(glob.glob(str(PROCESSED_DIR / 'master_panel_*.csv')))
    df = pd.read_csv(files[-1], parse_dates=['week_start'])
    return df


def build_segment_index(neighborhoods):
    """Compute composite segment_index per district."""
    rows = []
    # Normalise income to 0-1 scale
    incomes = [n['avg_income'] for n in neighborhoods.values()]
    income_min, income_max = min(incomes), max(incomes)

    for name, data in neighborhoods.items():
        income_norm = (data['avg_income'] - income_min) / (income_max - income_min)
        stores_per_100k = data['organic_stores'] / data['population'] * 100000

        segment_index = (
            0.30 * data['organic_affinity'] +
            0.25 * (data['green_voters_pct'] / 100) +
            0.20 * income_norm +
            0.15 * (1 - data['price_sensitivity']) +
            0.10 * min(stores_per_100k / 20, 1.0)
        )

        rows.append({
            'district': name,
            'segment_index': round(segment_index, 4),
            'organic_affinity': data['organic_affinity'],
            'green_voters_pct': data['green_voters_pct'],
            'income_norm': round(income_norm, 3),
            'price_sensitivity': data['price_sensitivity'],
            'stores_per_100k': round(stores_per_100k, 1),
            'primary_segment': data['primary_segment'],
        })

    return pd.DataFrame(rows)


def create_district_panel(panel, neighborhoods):
    """Expand city-wide panel to district-level pseudo-panel."""
    district_rows = []

    gt_cols = ['gt_norm_bio_tomaten', 'gt_norm_bio_salat', 'gt_norm_bio_gurken', 'gt_norm_bio_paprika']
    available_gt = [c for c in gt_cols if c in panel.columns]

    for _, row in panel.iterrows():
        for name, data in neighborhoods.items():
            district_row = {
                'week_start': row['week_start'],
                'district': name,
                'organic_affinity': data['organic_affinity'],
            }
            # Scale GT by organic affinity
            for col in available_gt:
                if pd.notna(row[col]):
                    district_row[f'{col}_district'] = row[col] * data['organic_affinity']
                else:
                    district_row[f'{col}_district'] = np.nan

            # Add controls
            for ctrl in ['temp_mean_weekly', 'food_price_index', 'consumer_confidence']:
                if ctrl in panel.columns:
                    district_row[ctrl] = row[ctrl]

            district_rows.append(district_row)

    return pd.DataFrame(district_rows)


def run_district_ols(district_panel, segment_df):
    """OLS: GT_norm_district ~ segment_index + controls."""
    # Merge segment index
    merged = district_panel.merge(segment_df[['district', 'segment_index']], on='district')
    merged = merged.dropna()

    gt_dist_cols = [c for c in merged.columns if c.endswith('_district')]
    if not gt_dist_cols:
        print("  No district GT columns found!")
        return pd.DataFrame()

    # Average across products for composite GT
    merged['gt_district_avg'] = merged[gt_dist_cols].mean(axis=1)

    # Create district dummies
    district_dummies = pd.get_dummies(merged['district'], drop_first=True, prefix='d', dtype=float)

    # Build regression
    y = merged['gt_district_avg'].astype(float)
    X_cols = ['segment_index']
    for ctrl in ['temp_mean_weekly', 'food_price_index']:
        if ctrl in merged.columns:
            X_cols.append(ctrl)

    X = merged[X_cols].astype(float).copy()
    X = pd.concat([X, district_dummies], axis=1)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    print(f"\n  R² = {model.rsquared:.4f}")
    print(f"  segment_index coef: {model.params.get('segment_index', 'N/A')}")
    print(f"  segment_index p-value: {model.pvalues.get('segment_index', 'N/A')}")

    # Results per district
    results = []
    for district in merged['district'].unique():
        seg_idx = segment_df[segment_df['district'] == district]['segment_index'].values[0]
        avg_gt = merged[merged['district'] == district]['gt_district_avg'].mean()
        results.append({
            'district': district,
            'segment_index': seg_idx,
            'avg_gt_district': round(avg_gt, 4),
            'segment_index_coef': round(model.params.get('segment_index', 0), 4),
            'segment_index_pvalue': round(model.pvalues.get('segment_index', 1), 6),
            'model_r2': round(model.rsquared, 4),
        })

    return pd.DataFrame(results)


def plot_berlin_choropleth(segment_df, results_df):
    """Stylized Berlin district map (matplotlib, no geopandas)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('H4: Berlin District Segment Intensity', fontsize=13, fontweight='bold')

    # Left: Segment index bar chart
    merged = segment_df.merge(results_df[['district', 'avg_gt_district']], on='district', how='left')
    merged = merged.sort_values('segment_index', ascending=True)

    colors = plt.cm.YlOrRd(merged['segment_index'] / merged['segment_index'].max())
    ax1.barh(merged['district'], merged['segment_index'], color=colors, edgecolor='white')
    ax1.set_xlabel('Segment Index')
    ax1.set_title('Composite Segment Index by District', loc='left')

    for i, (_, row) in enumerate(merged.iterrows()):
        ax1.text(row['segment_index'] + 0.01, i, f"{row['segment_index']:.3f}", va='center', fontsize=9)

    # Right: Segment index vs GT district scatter
    ax2.scatter(merged['segment_index'], merged['avg_gt_district'],
               s=merged['segment_index'] * 500, c=colors, edgecolors='black', linewidth=0.5, alpha=0.8)

    for _, row in merged.iterrows():
        ax2.annotate(row['district'], (row['segment_index'], row['avg_gt_district']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Regression line
    if len(merged) > 2:
        z = np.polyfit(merged['segment_index'], merged['avg_gt_district'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['segment_index'].min(), merged['segment_index'].max(), 100)
        ax2.plot(x_line, p(x_line), 'k--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('Avg GT Norm (District-adjusted)')
    ax2.set_title('Segment Index vs Search Intensity', loc='left')

    plt.tight_layout()
    filepath = FIGURE_DIR / 'H4_berlin_choropleth.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("H4 — District Segment Intensity")
    print("=" * 60)

    panel = load_panel()
    print(f"Panel: {len(panel)} weeks")

    # Build segment indices
    print("\n--- Segment Index Construction ---")
    segment_df = build_segment_index(NEIGHBORHOODS)
    print(segment_df[['district', 'segment_index', 'primary_segment']].to_string(index=False))

    # Create district-level pseudo-panel
    print("\n--- Creating District Panel ---")
    district_panel = create_district_panel(panel, NEIGHBORHOODS)
    print(f"District panel: {len(district_panel)} observations ({len(NEIGHBORHOODS)} districts x {len(panel)} weeks)")

    # Run OLS
    print("\n--- District OLS ---")
    results = run_district_ols(district_panel, segment_df)

    if not results.empty:
        filepath = TABLE_DIR / 'H4_district_coefficients.csv'
        results.to_csv(filepath, index=False)
        print(f"\nSaved: {filepath}")

    # Plot
    plot_berlin_choropleth(segment_df, results)

    # Verdict
    if not results.empty:
        p_val = results['segment_index_pvalue'].iloc[0]
        coef = results['segment_index_coef'].iloc[0]
        sig = p_val < 0.05
        print(f"\n{'='*60}")
        print(f"H4 VERDICT: {'SUPPORTED' if sig else 'NOT SUPPORTED'}")
        print(f"segment_index coefficient: {coef:.4f} (p={p_val:.6f})")
        print(f"Districts with higher segment index show {'stronger' if coef > 0 and sig else 'no significant difference in'} search intensity")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
