"""
AGRICOM - H5: Weather Distributed Lag Analysis
Tests: "Do weather conditions Granger-cause changes in organic produce search interest?"

Method: Distributed lag OLS with Newey-West HAC, Granger causality tests.
Verdict: >= 1 weather lag significant at p<0.05 for >= 2 of 4 products.

Input:  data/processed/master_panel_*.csv
Output: outputs/tables/H5_distributed_lag_coefficients.csv
        outputs/figures/H5_weather_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
FIGURE_DIR = PROJECT_DIR / 'outputs' / 'figures'
TABLE_DIR = PROJECT_DIR / 'outputs' / 'tables'

PRODUCTS = {
    'gt_norm_bio_tomaten': 'Tomaten',
    'gt_norm_bio_salat': 'Salat',
    'gt_norm_bio_gurken': 'Gurken',
    'gt_norm_bio_paprika': 'Paprika',
}

WEATHER_VARS = ['temp_mean_weekly', 'sunshine_hours_weekly']
MAX_LAG = 4


def load_panel():
    files = sorted(glob.glob(str(PROCESSED_DIR / 'master_panel_*.csv')))
    df = pd.read_csv(files[-1], parse_dates=['week_start'])
    return df.dropna(subset=list(PRODUCTS.keys()) + WEATHER_VARS)


def create_lag_features(df, weather_var, max_lag=MAX_LAG):
    """Create lag columns L0 through L{max_lag} for a weather variable."""
    result = df.copy()
    for lag in range(max_lag + 1):
        result[f'{weather_var}_L{lag}'] = result[weather_var].shift(lag)
    return result.dropna()


def run_distributed_lag_ols(df, product_col, weather_var, max_lag=MAX_LAG):
    """OLS: GT_norm ~ sum(beta_k * weather_lag_k) with Newey-West HAC."""
    lagged = create_lag_features(df, weather_var, max_lag)

    y = lagged[product_col]
    lag_cols = [f'{weather_var}_L{i}' for i in range(max_lag + 1)]
    X = sm.add_constant(lagged[lag_cols])

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})

    results = []
    for lag in range(max_lag + 1):
        col_name = f'{weather_var}_L{lag}'
        results.append({
            'product': product_col.replace('gt_norm_bio_', ''),
            'weather_var': weather_var,
            'lag': lag,
            'coefficient': model.params[col_name],
            'std_error': model.bse[col_name],
            't_stat': model.tvalues[col_name],
            'p_value': model.pvalues[col_name],
            'significant': model.pvalues[col_name] < 0.05,
        })

    return results, model


def run_granger_test(df, product_col, weather_var, max_lag=MAX_LAG):
    """Granger causality: does weather Granger-cause GT_norm?"""
    data = df[[product_col, weather_var]].dropna()
    if len(data) < 50:
        return None

    try:
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        granger_results = {}
        for lag in range(1, max_lag + 1):
            f_test = result[lag][0]['ssr_ftest']
            granger_results[lag] = {
                'f_stat': f_test[0],
                'p_value': f_test[1],
                'significant': f_test[1] < 0.05,
            }
        return granger_results
    except Exception as e:
        print(f"  Granger test failed for {product_col}/{weather_var}: {e}")
        return None


def plot_weather_scatter(df):
    """2x2 scatter grid: GT_norm vs temperature, color by month."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('H5: Organic Search Interest vs Temperature', fontsize=13, fontweight='bold')

    colors_map = {
        12: '#3498db', 1: '#3498db', 2: '#3498db',  # Winter blue
        3: '#2ecc71', 4: '#2ecc71', 5: '#2ecc71',   # Spring green
        6: '#e74c3c', 7: '#e74c3c', 8: '#e74c3c',   # Summer red
        9: '#f39c12', 10: '#f39c12', 11: '#f39c12',  # Autumn orange
    }

    for ax, (col, label) in zip(axes.flat, PRODUCTS.items()):
        data = df.dropna(subset=[col, 'temp_mean_weekly'])
        colors = data['month'].map(colors_map)

        ax.scatter(data['temp_mean_weekly'], data[col], c=colors, alpha=0.5, s=20)

        # Regression line
        z = np.polyfit(data['temp_mean_weekly'], data[col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['temp_mean_weekly'].min(), data['temp_mean_weekly'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.7)

        r, p_val = __import__('scipy').stats.pearsonr(data['temp_mean_weekly'], data[col])
        ax.set_title(f'{label} (r={r:.3f}, p={p_val:.3f})', fontsize=11)
        ax.set_xlabel('Weekly Mean Temp (°C)')
        ax.set_ylabel('GT Norm (0-1)')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Winter'),
                       Patch(facecolor='#2ecc71', label='Spring'),
                       Patch(facecolor='#e74c3c', label='Summer'),
                       Patch(facecolor='#f39c12', label='Autumn')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    filepath = FIGURE_DIR / 'H5_weather_scatter.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("H5 — Weather Distributed Lag Analysis")
    print("=" * 60)

    df = load_panel()
    print(f"Panel: {len(df)} complete weeks")

    all_results = []
    granger_summary = []

    for product_col, product_label in PRODUCTS.items():
        print(f"\n--- {product_label} ---")
        for weather_var in WEATHER_VARS:
            # Distributed lag OLS
            results, model = run_distributed_lag_ols(df, product_col, weather_var)
            all_results.extend(results)

            sig_lags = [r for r in results if r['significant']]
            print(f"  {weather_var}: {len(sig_lags)} significant lags, R²={model.rsquared:.3f}")
            for r in sig_lags:
                print(f"    Lag {r['lag']}: β={r['coefficient']:.4f}, p={r['p_value']:.4f}")

            # Granger test
            granger = run_granger_test(df, product_col, weather_var)
            if granger:
                for lag, res in granger.items():
                    granger_summary.append({
                        'product': product_col.replace('gt_norm_bio_', ''),
                        'weather_var': weather_var,
                        'granger_lag': lag,
                        'f_stat': res['f_stat'],
                        'p_value': res['p_value'],
                        'significant': res['significant'],
                    })

    # Save results
    results_df = pd.DataFrame(all_results)
    filepath = TABLE_DIR / 'H5_distributed_lag_coefficients.csv'
    results_df.to_csv(filepath, index=False)
    print(f"\nSaved: {filepath}")

    if granger_summary:
        granger_df = pd.DataFrame(granger_summary)
        granger_path = TABLE_DIR / 'H5_granger_causality.csv'
        granger_df.to_csv(granger_path, index=False)
        print(f"Saved: {granger_path}")

    # Plot
    plot_weather_scatter(df)

    # Verdict
    products_with_sig = set()
    for r in all_results:
        if r['significant']:
            products_with_sig.add(r['product'])

    verdict = len(products_with_sig) >= 2
    print(f"\n{'='*60}")
    print(f"H5 VERDICT: {'SUPPORTED' if verdict else 'NOT SUPPORTED'}")
    print(f"Products with significant weather lags: {products_with_sig}")
    print(f"Criterion: >= 2 products needed, found {len(products_with_sig)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
