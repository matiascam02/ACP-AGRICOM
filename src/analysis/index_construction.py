"""
AGRICOM - Phase 4: Index Construction & Prototype
Constructs product-level D(p,w) and basket composite BasketIndex(w).
Includes retrospective event validation and weight sensitivity.

Input:  data/processed/master_panel_*.csv
Output: outputs/figures/product_index_*.png
        outputs/figures/basket_index_timeseries.png
        outputs/figures/tomato_weight_sensitivity.png
        outputs/data/product_indices.csv
        outputs/tables/basket_product_correlations.csv
        outputs/tables/retrospective_validation.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
RAW_PRICING = PROJECT_DIR / 'data' / 'raw' / 'pricing'
FIGURE_DIR = PROJECT_DIR / 'outputs' / 'figures'
TABLE_DIR = PROJECT_DIR / 'outputs' / 'tables'
DATA_DIR = PROJECT_DIR / 'outputs' / 'data'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

PRODUCTS = {
    'tomaten': {'gt_col': 'gt_norm_bio_tomaten', 'color': '#e74c3c', 'label': 'Bio Tomaten', 'basket_w': 0.35},
    'salat': {'gt_col': 'gt_norm_bio_salat', 'color': '#2ecc71', 'label': 'Bio Salat', 'basket_w': 0.30},
    'gurken': {'gt_col': 'gt_norm_bio_gurken', 'color': '#27ae60', 'label': 'Bio Gurken', 'basket_w': 0.20},
    'paprika': {'gt_col': 'gt_norm_bio_paprika', 'color': '#f39c12', 'label': 'Bio Paprika', 'basket_w': 0.15},
}

# Component weights for D(p,w)
ALPHA = 0.5   # GT weight
BETA = 0.3    # Price premium weight
GAMMA = 0.2   # Weather weight

# Known events for retrospective validation
KNOWN_EVENTS = {
    'Fruit Logistica 2023': pd.Timestamp('2023-02-06'),
    'Fruit Logistica 2024': pd.Timestamp('2024-02-05'),
    'Fruit Logistica 2025': pd.Timestamp('2025-02-03'),
    'Summer Peak 2023': pd.Timestamp('2023-07-10'),
    'Summer Peak 2024': pd.Timestamp('2024-07-08'),
    'Summer Peak 2025': pd.Timestamp('2025-07-07'),
    'Christmas 2023': pd.Timestamp('2023-12-18'),
    'Christmas 2024': pd.Timestamp('2024-12-16'),
    'Easter 2023': pd.Timestamp('2023-04-03'),
    'Easter 2024': pd.Timestamp('2024-03-25'),
    'Easter 2025': pd.Timestamp('2025-04-14'),
    'Heatwave 2023': pd.Timestamp('2023-07-17'),
    'Winter Trough 2023': pd.Timestamp('2023-01-16'),
    'Winter Trough 2024': pd.Timestamp('2024-01-15'),
}

# Base prices for illustrative RP_norm when AMI not available
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


def compute_weather_norm(df):
    """Rank-normalise temperature over full panel."""
    temp = df['temp_mean_weekly'].values
    valid_mask = ~np.isnan(temp)
    weather_norm = np.full_like(temp, np.nan)
    weather_norm[valid_mask] = rankdata(temp[valid_mask]) / valid_mask.sum()
    return weather_norm


def compute_rp_norm(df, product):
    """Compute RP_norm from food_price_index as illustrative proxy."""
    if 'food_price_index' not in df.columns:
        return np.full(len(df), np.nan)

    fpi = df['food_price_index'].values
    base = BASE_PRICES[product]
    baseline_fpi = fpi[~np.isnan(fpi)][0] if any(~np.isnan(fpi)) else 100

    months = df['month'].values
    seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * (months - 3) / 12)

    organic = base['organic'] * (fpi / baseline_fpi) * seasonal
    conventional = base['conventional'] * (fpi / baseline_fpi)
    premium = (organic - conventional) / conventional

    # RP_norm: higher = smaller premium = more demand expected
    max_p = np.nanmax(premium)
    min_p = np.nanmin(premium)
    if max_p == min_p:
        return np.full(len(df), 0.5)
    return np.clip((max_p - premium) / (max_p - min_p), 0, 1)


def compute_product_index(df, product, gt_weight=ALPHA, rp_weight=BETA, weather_weight=GAMMA):
    """Compute D(p,w) = alpha*GT_norm + beta*RP_norm + gamma*Weather_norm."""
    config = PRODUCTS[product]
    gt_norm = df[config['gt_col']].values

    weather_norm = compute_weather_norm(df)
    rp_norm = compute_rp_norm(df, product)

    # Check if RP_norm is all NaN
    has_prices = not np.all(np.isnan(rp_norm))

    if has_prices:
        d = gt_weight * gt_norm + rp_weight * rp_norm + weather_weight * weather_norm
    else:
        # Reweight without prices
        total = gt_weight + weather_weight
        d = (gt_weight / total) * gt_norm + (weather_weight / total) * weather_norm

    return d * 100  # Scale to 0-100


def plot_product_indices(df, indices):
    """2x2 subplot: D(p,w) over time per product."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle('Product-Level Demand Index D(p,w)', fontsize=13, fontweight='bold')

    for ax, (product, series) in zip(axes.flat, indices.items()):
        config = PRODUCTS[product]
        ax.plot(df['week_start'], series, color=config['color'], linewidth=1, alpha=0.7)

        # 8-week rolling average
        rolling = pd.Series(series).rolling(8, center=True).mean()
        ax.plot(df['week_start'], rolling, color=config['color'], linewidth=2)

        ax.set_title(config['label'], fontsize=11, fontweight='bold', loc='left')
        ax.set_ylabel('D(p,w)')
        ax.set_ylim(0, 100)

    plt.tight_layout()
    filepath = FIGURE_DIR / 'product_index_all.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def compute_basket_index(indices):
    """BasketIndex(w) = weighted sum of product D(p,w). Preserves NaN where any product is NaN."""
    stacked = np.column_stack(list(indices.values()))
    any_nan = np.any(np.isnan(stacked), axis=1)

    basket = np.zeros(len(stacked))
    for i, (product, series) in enumerate(indices.items()):
        w = PRODUCTS[product]['basket_w']
        basket += w * np.nan_to_num(series, nan=0)

    basket[any_nan] = np.nan
    return basket


def plot_basket_index(df, basket_index, indices):
    """Basket index time series with product overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Organic Produce Demand Index — Berlin Basket Composite', fontsize=14, fontweight='bold')

    # Top: basket index
    ax1.plot(df['week_start'], basket_index, color='#2c3e50', linewidth=1.5, label='Basket Index')
    rolling = pd.Series(basket_index).rolling(8, center=True).mean()
    ax1.plot(df['week_start'], rolling, color='#2c3e50', linewidth=2.5, alpha=0.5, label='8-week MA')

    ax1.set_ylabel('BasketIndex(w)')
    ax1.set_title(f'Weights: Tomaten {PRODUCTS["tomaten"]["basket_w"]}, '
                  f'Salat {PRODUCTS["salat"]["basket_w"]}, '
                  f'Gurken {PRODUCTS["gurken"]["basket_w"]}, '
                  f'Paprika {PRODUCTS["paprika"]["basket_w"]}', loc='left', fontsize=10)
    ax1.legend(loc='upper right')

    # Bottom: individual products (faded)
    for product, series in indices.items():
        config = PRODUCTS[product]
        ax2.plot(df['week_start'], series, color=config['color'], linewidth=0.8, alpha=0.5, label=config['label'])
    ax2.set_ylabel('D(p,w)')
    ax2.set_xlabel('Week')
    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    filepath = FIGURE_DIR / 'basket_index_timeseries.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def basket_product_correlations(df, basket_index, indices):
    """Correlation of basket index with each product D(p,w)."""
    from scipy import stats

    rows = []
    for product, series in indices.items():
        valid = ~np.isnan(series) & ~np.isnan(basket_index)
        if valid.sum() > 10:
            r, p = stats.pearsonr(basket_index[valid], series[valid])
            rows.append({'product': product, 'pearson_r': round(r, 4), 'p_value': round(p, 6)})

    corr_df = pd.DataFrame(rows)
    filepath = TABLE_DIR / 'basket_product_correlations.csv'
    corr_df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")
    return corr_df


def retrospective_validation(df, basket_index):
    """Identify 3 highest + 3 lowest weeks. Cross-reference with known events."""
    bi_series = pd.Series(basket_index, index=df['week_start'])
    bi_series = bi_series.dropna()

    top3 = bi_series.nlargest(3)
    bottom3 = bi_series.nsmallest(3)

    rows = []
    for date, value in list(top3.items()) + list(bottom3.items()):
        rank_type = 'high' if date in top3.index else 'low'

        # Find closest known event (within 3 weeks)
        matched_event = None
        min_dist = float('inf')
        for event_name, event_date in KNOWN_EVENTS.items():
            dist = abs((date - event_date).days)
            if dist < min_dist:
                min_dist = dist
                if dist <= 21:
                    matched_event = event_name

        rows.append({
            'week_date': date.strftime('%Y-%m-%d'),
            'basket_value': round(value, 2),
            'rank_type': rank_type,
            'matched_event': matched_event or 'No match',
            'plausible': matched_event is not None,
        })

    result_df = pd.DataFrame(rows)
    filepath = TABLE_DIR / 'retrospective_validation.csv'
    result_df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")

    n_plausible = result_df['plausible'].sum()
    print(f"\nRetrospective validation: {n_plausible}/6 extreme weeks matched known events")
    print(result_df.to_string(index=False))
    return result_df


def tomato_weight_sensitivity(df):
    """3 weight specifications for tomato D(p,w)."""
    specs = {
        'Baseline (0.5/0.3/0.2)': (0.5, 0.3, 0.2),
        'GT-Heavy (0.7/0.15/0.15)': (0.7, 0.15, 0.15),
        'Price-Heavy (0.3/0.5/0.2)': (0.3, 0.5, 0.2),
    }

    colors = ['#2c3e50', '#e74c3c', '#3498db']

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle('Weight Sensitivity — Tomato D(p,w) Under 3 Specifications', fontsize=13, fontweight='bold')

    for (name, (a, b, g)), color in zip(specs.items(), colors):
        series = compute_product_index(df, 'tomaten', gt_weight=a, rp_weight=b, weather_weight=g)
        rolling = pd.Series(series).rolling(8, center=True).mean()
        ax.plot(df['week_start'], rolling, label=name, color=color, linewidth=1.5)

    ax.set_ylabel('D(tomato, w)')
    ax.set_xlabel('Week')
    ax.legend(loc='upper right')

    plt.tight_layout()
    filepath = FIGURE_DIR / 'tomato_weight_sensitivity.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("Phase 4 — Index Construction & Prototype")
    print("=" * 60)

    df = load_panel()
    print(f"Panel: {len(df)} weeks")

    # 4.1 Product-level indices
    print("\n--- 4.1 Product-Level Index D(p,w) ---")
    indices = {}
    for product in PRODUCTS:
        indices[product] = compute_product_index(df, product)
        valid = ~np.isnan(indices[product])
        print(f"  {product}: mean={np.nanmean(indices[product]):.1f}, "
              f"std={np.nanstd(indices[product]):.1f}, "
              f"valid={valid.sum()}/{len(df)}")

    plot_product_indices(df, indices)

    # Save product indices
    indices_df = df[['week_start']].copy()
    for product, series in indices.items():
        indices_df[f'D_{product}'] = series
    indices_df.to_csv(DATA_DIR / 'product_indices.csv', index=False)
    print(f"Saved: {DATA_DIR / 'product_indices.csv'}")

    # 4.2 Basket composite
    print("\n--- 4.2 Basket Composite ---")
    basket_index = compute_basket_index(indices)
    print(f"  BasketIndex: mean={np.nanmean(basket_index):.1f}, std={np.nanstd(basket_index):.1f}")
    plot_basket_index(df, basket_index, indices)

    print("\n  Product-Basket Correlations:")
    corr_df = basket_product_correlations(df, basket_index, indices)
    print(corr_df.to_string(index=False))

    # 4.3 Retrospective validation
    print("\n--- 4.3 Retrospective Event Validation ---")
    validation = retrospective_validation(df, basket_index)

    n_plausible = validation['plausible'].sum()
    verdict = n_plausible >= 4

    # 4.4 Weight sensitivity
    print("\n--- 4.4 Weight Sensitivity (Tomato) ---")
    tomato_weight_sensitivity(df)

    # Save basket index
    basket_df = df[['week_start']].copy()
    basket_df['basket_index'] = basket_index
    basket_df.to_csv(DATA_DIR / 'basket_index.csv', index=False)
    print(f"Saved: {DATA_DIR / 'basket_index.csv'}")

    print(f"\n{'='*60}")
    print("Phase 4 COMPLETE")
    print(f"Retrospective validation: {'PASSED' if verdict else 'PARTIAL'} ({n_plausible}/6 matches, need >= 4)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
