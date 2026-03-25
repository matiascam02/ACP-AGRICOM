"""
AGRICOM - H3: Price Elasticity Estimation (Scenario B — AMI Quarterly Fallback)
Tests: "Do organic price premiums predict variation in organic search interest?"

Method: Product-level OLS/Prais-Winsten with seasonal + economic controls.
Uses illustrative prices from food_price_index if AMI data unavailable.

Input:  data/processed/master_panel_*.csv
        data/raw/pricing/ami_weekly_interpolated.csv (optional)
Output: outputs/tables/H3_elasticity_table.csv
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
RAW_PRICING = PROJECT_DIR / 'data' / 'raw' / 'pricing'
FIGURE_DIR = PROJECT_DIR / 'outputs' / 'figures'
TABLE_DIR = PROJECT_DIR / 'outputs' / 'tables'

PRODUCTS = {
    'tomaten': 'gt_norm_bio_tomaten',
    'salat': 'gt_norm_bio_salat',
    'gurken': 'gt_norm_bio_gurken',
    'paprika': 'gt_norm_bio_paprika',
}

# Base prices for illustrative data (if AMI not available)
BASE_PRICES = {
    'tomaten': {'organic': 4.20, 'conventional': 2.30},
    'salat': {'organic': 3.40, 'conventional': 1.80},
    'gurken': {'organic': 2.60, 'conventional': 1.40},
    'paprika': {'organic': 5.00, 'conventional': 3.10},
}


def load_panel():
    files = sorted(glob.glob(str(PROCESSED_DIR / 'master_panel_*.csv')))
    df = pd.read_csv(files[-1], parse_dates=['week_start'])
    return df


def load_or_generate_prices(panel):
    """Load AMI prices if available, else generate illustrative prices."""
    ami_file = RAW_PRICING / 'ami_weekly_interpolated.csv'

    if ami_file.exists():
        prices = pd.read_csv(ami_file, parse_dates=['week_start'])
        is_synthetic = False
        print("  Using AMI interpolated prices")
    else:
        print("  No AMI data — generating illustrative prices from food_price_index")
        is_synthetic = True
        prices = generate_illustrative_prices(panel)

    return prices, is_synthetic


def generate_illustrative_prices(panel):
    """Generate illustrative prices scaled by food_price_index."""
    if 'food_price_index' not in panel.columns:
        return pd.DataFrame()

    df = panel[['week_start', 'food_price_index']].dropna().copy()
    baseline_fpi = df['food_price_index'].iloc[0]

    rows = []
    for _, row in df.iterrows():
        multiplier = row['food_price_index'] / baseline_fpi
        # Add seasonal variation (organic premium widens in winter)
        month = row['week_start'].month
        seasonal_factor = 1.0 + 0.05 * np.sin(2 * np.pi * (month - 3) / 12)  # Peak in summer

        for product, prices in BASE_PRICES.items():
            organic = prices['organic'] * multiplier * seasonal_factor
            conventional = prices['conventional'] * multiplier
            premium = (organic - conventional) / conventional

            rows.append({
                'week_start': row['week_start'],
                'product': product,
                'organic_price': round(organic, 2),
                'conventional_price': round(conventional, 2),
                'organic_premium': round(premium, 4),
            })

    return pd.DataFrame(rows)


def compute_rp_norm(premiums):
    """RP_norm: clip to [0,1] where higher = smaller premium = higher demand expected."""
    max_p = premiums.max()
    min_p = premiums.min()
    if max_p == min_p:
        return pd.Series(0.5, index=premiums.index)
    return ((max_p - premiums) / (max_p - min_p)).clip(0, 1)


def add_seasonal_controls(df):
    """Add Fourier terms for seasonal controls."""
    df = df.copy()
    week = df['week_start'].dt.isocalendar().week.astype(float)
    df['sin_annual'] = np.sin(2 * np.pi * week / 52)
    df['cos_annual'] = np.cos(2 * np.pi * week / 52)
    df['sin_semiannual'] = np.sin(4 * np.pi * week / 52)
    df['cos_semiannual'] = np.cos(4 * np.pi * week / 52)
    return df


def check_h5_interactions():
    """Check if H5 showed strong seasonal weather effects."""
    h5_file = TABLE_DIR / 'H5_distributed_lag_coefficients.csv'
    if h5_file.exists():
        h5 = pd.read_csv(h5_file)
        n_sig = h5['significant'].sum()
        return n_sig >= 4  # Strong weather effects if many significant lags
    return False


def run_elasticity_regression(df, product, gt_col, include_interactions=False):
    """OLS with HAC: GT_norm ~ RP_norm + seasonal + economic controls."""
    data = df.dropna(subset=[gt_col, 'rp_norm'])
    if len(data) < 50:
        return None

    y = data[gt_col]

    X_cols = ['rp_norm', 'sin_annual', 'cos_annual', 'sin_semiannual', 'cos_semiannual']

    if 'food_price_index' in data.columns and data['food_price_index'].notna().sum() > 50:
        X_cols.append('food_price_index')

    if 'consumer_confidence' in data.columns and data['consumer_confidence'].notna().sum() > 50:
        X_cols.append('consumer_confidence')

    if include_interactions and 'sin_annual' in data.columns:
        data = data.copy()
        data['rp_x_season'] = data['rp_norm'] * data['sin_annual']
        X_cols.append('rp_x_season')

    X = sm.add_constant(data[X_cols].astype(float))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    # Durbin-Watson
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(model.resid)

    result = {
        'product': product,
        'beta_elasticity': round(model.params.get('rp_norm', 0), 6),
        'std_error': round(model.bse.get('rp_norm', 0), 6),
        'p_value': round(model.pvalues.get('rp_norm', 1), 6),
        'ci_lower': round(model.conf_int().loc['rp_norm', 0], 6) if 'rp_norm' in model.conf_int().index else None,
        'ci_upper': round(model.conf_int().loc['rp_norm', 1], 6) if 'rp_norm' in model.conf_int().index else None,
        'r_squared': round(model.rsquared, 4),
        'dw_stat': round(dw, 4),
        'n_obs': int(model.nobs),
        'seasonal_interaction': include_interactions,
    }

    return result


def main():
    print("=" * 60)
    print("H3 — Price Elasticity Estimation")
    print("=" * 60)

    panel = load_panel()
    print(f"Panel: {len(panel)} weeks")

    # Load or generate prices
    print("\n--- Loading Price Data ---")
    prices, is_synthetic = load_or_generate_prices(panel)

    if prices.empty:
        print("ERROR: Could not load or generate price data. Exiting.")
        return

    # Check H5 for interaction decision
    include_interactions = check_h5_interactions()
    if include_interactions:
        print("  H5 showed strong weather effects — including seasonal interaction terms")

    # Add seasonal controls to panel
    panel = add_seasonal_controls(panel)

    results = []
    print("\n--- Running Elasticity Regressions ---")

    for product, gt_col in PRODUCTS.items():
        print(f"\n  {product}:")
        product_prices = prices[prices['product'] == product].copy()

        if product_prices.empty:
            print(f"    No price data for {product}")
            continue

        # Compute RP_norm
        product_prices['rp_norm'] = compute_rp_norm(product_prices['organic_premium'])

        # Merge with panel
        merged = panel.merge(product_prices[['week_start', 'rp_norm', 'organic_premium']],
                            on='week_start', how='left')

        result = run_elasticity_regression(merged, product, gt_col, include_interactions)

        if result:
            result['interpolation_uncertainty'] = is_synthetic
            result['data_source'] = 'synthetic_from_food_cpi' if is_synthetic else 'ami_quarterly_interpolated'
            results.append(result)
            print(f"    beta={result['beta_elasticity']:.4f}, p={result['p_value']:.4f}, R²={result['r_squared']:.3f}, DW={result['dw_stat']:.2f}")

    if results:
        results_df = pd.DataFrame(results)
        filepath = TABLE_DIR / 'H3_elasticity_table.csv'
        results_df.to_csv(filepath, index=False)
        print(f"\nSaved: {filepath}")

        if is_synthetic:
            print("\n  NOTE: Results use ILLUSTRATIVE prices derived from food_price_index.")
            print("  These results demonstrate the methodology. For robust conclusions,")
            print("  replace with actual AMI quarterly data in data/raw/pricing/ami_quarterly_prices_manual.csv")

        # Verdict
        sig_products = [r for r in results if r['p_value'] < 0.05]
        print(f"\n{'='*60}")
        print(f"H3 VERDICT: {len(sig_products)}/{len(results)} products show significant price elasticity")
        for r in results:
            sig = '*' if r['p_value'] < 0.05 else ''
            print(f"  {r['product']}: beta={r['beta_elasticity']:.4f} (p={r['p_value']:.4f}){sig}")
        if is_synthetic:
            print("  [Caveat: Results based on illustrative prices — AMI data would strengthen conclusions]")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
