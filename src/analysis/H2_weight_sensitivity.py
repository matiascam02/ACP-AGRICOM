"""
AGRICOM - H2: Basket Weighting Validity
Tests: "Does the 35/30/20/15 weighting produce a meaningfully different index than alternatives?"

Method: Compare 3 weight specifications via MAD and Spearman rank concordance.
Verdict: If pairwise rho > 0.90 or MAD < 5, weights are robust.

Input:  data/processed/master_panel_*.csv
Output: outputs/tables/H2_weight_comparison.csv
        outputs/figures/H2_three_index_sensitivity.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
FIGURE_DIR = PROJECT_DIR / 'outputs' / 'figures'
TABLE_DIR = PROJECT_DIR / 'outputs' / 'tables'

PRODUCT_COLS = ['gt_norm_bio_tomaten', 'gt_norm_bio_salat', 'gt_norm_bio_gurken', 'gt_norm_bio_paprika']

WEIGHT_SPECS = {
    'established': [0.35, 0.30, 0.20, 0.15],
    'equal': [0.25, 0.25, 0.25, 0.25],
    'bmel_calibrated': [0.40, 0.25, 0.20, 0.15],
}


def load_panel():
    files = sorted(glob.glob(str(PROCESSED_DIR / 'master_panel_*.csv')))
    df = pd.read_csv(files[-1], parse_dates=['week_start'])
    return df.dropna(subset=PRODUCT_COLS)


def compute_basket_index(df, weights):
    """Compute weighted basket index from GT_norm columns."""
    index = np.zeros(len(df))
    for col, w in zip(PRODUCT_COLS, weights):
        index += w * df[col].values
    return pd.Series(index * 100, index=df.index, name='basket_index')  # Scale to 0-100


def compare_indices(indices):
    """Pairwise MAD and Spearman comparison."""
    names = list(indices.keys())
    results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1 = indices[names[i]]
            s2 = indices[names[j]]

            mad = np.mean(np.abs(s1 - s2))
            rho, p = stats.spearmanr(s1, s2)

            results.append({
                'variant_1': names[i],
                'variant_2': names[j],
                'MAD': round(mad, 3),
                'spearman_rho': round(rho, 4),
                'spearman_p': round(p, 8),
                'robust': rho > 0.90 or mad < 5,
            })

    return pd.DataFrame(results)


def plot_three_indices(df, indices):
    """Plot all 3 index variants on same time axis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('H2: Basket Index — Weight Sensitivity (3 Specifications)', fontsize=13, fontweight='bold')

    colors = {'established': '#2c3e50', 'equal': '#e74c3c', 'bmel_calibrated': '#3498db'}

    for name, series in indices.items():
        ax1.plot(df['week_start'], series, label=f'{name} ({WEIGHT_SPECS[name]})',
                color=colors[name], linewidth=1.2, alpha=0.8)

    ax1.set_ylabel('Basket Index (0-100)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('Index Variants Over Time', loc='left')

    # Difference from established
    for name in ['equal', 'bmel_calibrated']:
        diff = indices[name] - indices['established']
        ax2.plot(df['week_start'], diff, label=f'{name} - established',
                color=colors[name], linewidth=1, alpha=0.7)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Difference')
    ax2.set_xlabel('Week')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('Deviation from Established Weights', loc='left')

    plt.tight_layout()
    filepath = FIGURE_DIR / 'H2_three_index_sensitivity.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("H2 — Basket Weighting Validity")
    print("=" * 60)

    df = load_panel()
    print(f"Panel: {len(df)} complete weeks")

    # Compute 3 variants
    indices = {}
    for name, weights in WEIGHT_SPECS.items():
        indices[name] = compute_basket_index(df, weights)
        print(f"  {name}: mean={indices[name].mean():.2f}, std={indices[name].std():.2f}")

    # Compare
    print("\n--- Pairwise Comparison ---")
    comparison = compare_indices(indices)
    print(comparison.to_string(index=False))

    filepath = TABLE_DIR / 'H2_weight_comparison.csv'
    comparison.to_csv(filepath, index=False)
    print(f"\nSaved: {filepath}")

    # Plot
    plot_three_indices(df, indices)

    # Verdict
    all_robust = comparison['robust'].all()
    print(f"\n{'='*60}")
    print(f"H2 VERDICT: {'WEIGHTS ROBUST' if all_robust else 'WEIGHTS MATTER'}")
    for _, row in comparison.iterrows():
        print(f"  {row['variant_1']} vs {row['variant_2']}: MAD={row['MAD']:.2f}, rho={row['spearman_rho']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
