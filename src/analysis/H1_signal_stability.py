"""
AGRICOM - H1: Search Signal Stability
Tests: "Is Google Trends search interest for organic produce a stable, interpretable signal?"

H1a: ACF/PACF — meaningful seasonal structure at 52-week lag.
H1b: Spearman split-half — rank correlation > 0.60 between halves.

Input:  data/processed/master_panel_*.csv
Output: outputs/figures/H1a_acf_<product>.png
        outputs/tables/H1b_spearman_stability.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
FIGURE_DIR = PROJECT_DIR / 'outputs' / 'figures'
TABLE_DIR = PROJECT_DIR / 'outputs' / 'tables'

PRODUCTS = {
    'gt_bio_tomaten': 'Tomaten',
    'gt_bio_salat': 'Salat',
    'gt_bio_gurken': 'Gurken',
    'gt_bio_paprika': 'Paprika',
}

NLAGS = 60  # Show up to 60 lags to capture 52-week seasonal


def load_panel():
    files = sorted(glob.glob(str(PROCESSED_DIR / 'master_panel_*.csv')))
    df = pd.read_csv(files[-1], parse_dates=['week_start'])
    return df


def plot_acf_pacf(series, product_name, product_label):
    """Compute and plot ACF/PACF with 52-week lag highlighted."""
    clean = series.dropna().values
    if len(clean) < NLAGS + 10:
        print(f"  Skipping {product_name}: insufficient data")
        return None

    acf_values, confint_acf = acf(clean, nlags=NLAGS, alpha=0.05)
    pacf_values, confint_pacf = pacf(clean, nlags=min(NLAGS, len(clean)//2 - 1), alpha=0.05)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'H1a: ACF/PACF — {product_label}', fontsize=13, fontweight='bold')

    # ACF
    lags = np.arange(len(acf_values))
    ax1.bar(lags, acf_values, width=0.8, color='steelblue', alpha=0.7)
    ax1.fill_between(lags, confint_acf[:, 0] - acf_values, confint_acf[:, 1] - acf_values,
                      alpha=0.2, color='gray')
    ax1.axhline(0, color='black', linewidth=0.5)
    if len(acf_values) > 52:
        ax1.axvline(52, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='52-week lag')
        ax1.annotate(f'Lag 52: {acf_values[52]:.3f}', xy=(52, acf_values[52]),
                    xytext=(55, acf_values[52] + 0.1), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red'), color='red')
    ax1.set_ylabel('ACF')
    ax1.set_title('Autocorrelation Function', loc='left')
    ax1.legend()

    # PACF
    pacf_lags = np.arange(len(pacf_values))
    ax2.bar(pacf_lags, pacf_values, width=0.8, color='darkorange', alpha=0.7)
    ax2.fill_between(pacf_lags, confint_pacf[:, 0] - pacf_values, confint_pacf[:, 1] - pacf_values,
                      alpha=0.2, color='gray')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('PACF')
    ax2.set_xlabel('Lag (weeks)')
    ax2.set_title('Partial Autocorrelation Function', loc='left')

    plt.tight_layout()
    filepath = FIGURE_DIR / f'H1a_acf_{product_name}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

    return acf_values


def spearman_split_half(df, product_col):
    """Split panel at midpoint. Compare week-of-year seasonal profiles via Spearman rho."""
    series = df[['week_start', 'week_of_year', product_col]].dropna()

    midpoint = series['week_start'].median()
    first_half = series[series['week_start'] <= midpoint]
    second_half = series[series['week_start'] > midpoint]

    # Compute week-of-year median profile for each half
    profile_1 = first_half.groupby('week_of_year')[product_col].median()
    profile_2 = second_half.groupby('week_of_year')[product_col].median()

    # Align on common weeks
    common_weeks = profile_1.index.intersection(profile_2.index)
    if len(common_weeks) < 20:
        return {'spearman_rho': np.nan, 'p_value': np.nan, 'stable': False,
                'n_weeks_half1': len(first_half), 'n_weeks_half2': len(second_half)}

    rho, p_value = stats.spearmanr(profile_1.loc[common_weeks], profile_2.loc[common_weeks])

    return {
        'product': product_col.replace('gt_bio_', ''),
        'spearman_rho': round(rho, 4),
        'p_value': round(p_value, 6),
        'stable': rho > 0.60,
        'n_weeks_half1': len(first_half),
        'n_weeks_half2': len(second_half),
        'n_common_weeks': len(common_weeks),
    }


def main():
    print("=" * 60)
    print("H1 — Search Signal Stability")
    print("=" * 60)

    df = load_panel()

    # H1a: ACF/PACF
    print("\n--- H1a: ACF/PACF Analysis ---")
    acf_results = {}
    for col, label in PRODUCTS.items():
        print(f"\n{label}:")
        series = df.set_index('week_start')[col]
        acf_vals = plot_acf_pacf(series, col.replace('gt_bio_', ''), label)
        if acf_vals is not None and len(acf_vals) > 52:
            acf_results[col] = acf_vals[52]
            print(f"  ACF at lag 52: {acf_vals[52]:.3f}")

    # H1b: Spearman split-half
    print("\n--- H1b: Spearman Split-Half Stability ---")
    stability_results = []
    for col, label in PRODUCTS.items():
        result = spearman_split_half(df, col)
        stability_results.append(result)
        print(f"  {label}: rho={result['spearman_rho']:.4f}, p={result['p_value']:.6f}, "
              f"stable={'YES' if result['stable'] else 'NO'}")

    stability_df = pd.DataFrame(stability_results)
    filepath = TABLE_DIR / 'H1b_spearman_stability.csv'
    stability_df.to_csv(filepath, index=False)
    print(f"\nSaved: {filepath}")

    # Verdict
    n_stable = sum(1 for r in stability_results if r.get('stable', False))
    print(f"\n{'='*60}")
    print(f"H1 VERDICT: {n_stable}/{len(PRODUCTS)} products show stable seasonal patterns (rho > 0.60)")
    for col, label in PRODUCTS.items():
        acf52 = acf_results.get(col, 'N/A')
        print(f"  {label}: ACF(52) = {acf52:.3f}" if isinstance(acf52, float) else f"  {label}: ACF(52) = {acf52}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
