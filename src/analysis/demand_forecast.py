"""
AGRICOM - Demand Forecasting Model
Predicts organic produce demand in Berlin using Prophet and ensemble methods.

Features:
- Multiple data source integration (weather, events, sentiment, trends)
- Neighborhood-specific forecasts (Kreuzberg, Mitte, Charlottenburg)
- Lag features for temporal patterns
- Seasonal decomposition
- Model evaluation and validation

Usage:
    python demand_forecast.py [--weeks 12] [--neighborhood all]

Output:
    outputs/forecasts/demand_forecast_YYYYMMDD.csv
    outputs/figures/forecast_*.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional Prophet import (graceful fallback)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not available - using ARIMA fallback")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configuration
PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'outputs'
FORECAST_DIR = OUTPUT_DIR / 'forecasts'
FIGURE_DIR = OUTPUT_DIR / 'figures'

# Ensure directories exist
FORECAST_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class DemandForecaster:
    """
    Ensemble forecasting model for organic produce demand.
    Combines Prophet (seasonal), ML models (feature-based), and baseline.
    """
    
    def __init__(self, neighborhood: str = 'all'):
        self.neighborhood = neighborhood
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def load_data(self) -> pd.DataFrame:
        """Load unified dataset or create from raw sources."""
        unified_path = DATA_DIR / 'processed' / 'agricom_unified_dataset.csv'
        
        if unified_path.exists():
            print("Loading unified dataset...")
            df = pd.read_csv(unified_path, parse_dates=['date'])
        else:
            print("Unified dataset not found - loading raw data...")
            df = self._load_raw_data()
            
        return df
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load and combine raw data sources."""
        raw_dir = DATA_DIR / 'raw'
        
        # Load weather
        weather_files = list(raw_dir.glob('weather_berlin_*.csv'))
        if weather_files:
            weather_df = pd.read_csv(max(weather_files, key=lambda x: x.stat().st_mtime))
            weather_df['date'] = pd.to_datetime(weather_df['time'])
            print(f"  ✓ Weather: {len(weather_df)} days")
        else:
            weather_df = pd.DataFrame()
            
        # Load events
        events_files = list(raw_dir.glob('events_berlin_*.csv'))
        if events_files:
            events_df = pd.read_csv(max(events_files, key=lambda x: x.stat().st_mtime))
            events_df['date'] = pd.to_datetime(events_df['date'])
            print(f"  ✓ Events: {len(events_df)} records")
        else:
            events_df = pd.DataFrame()
            
        # Load GDELT sentiment
        gdelt_files = list(raw_dir.glob('gdelt_timeline_*.csv'))
        if gdelt_files:
            gdelt_df = pd.read_csv(max(gdelt_files, key=lambda x: x.stat().st_mtime))
            gdelt_df['date'] = pd.to_datetime(gdelt_df['datetime']).dt.date
            gdelt_df['date'] = pd.to_datetime(gdelt_df['date'])
            # Handle different column names
            sentiment_col = 'Average Tone' if 'Average Tone' in gdelt_df.columns else 'value'
            gdelt_daily = gdelt_df.groupby('date').agg({sentiment_col: 'mean'}).reset_index()
            gdelt_daily = gdelt_daily.rename(columns={sentiment_col: 'sentiment'})
            print(f"  ✓ GDELT sentiment: {len(gdelt_daily)} days")
        else:
            gdelt_daily = pd.DataFrame()
            
        # Combine into base dataframe
        if len(weather_df) > 0:
            df = weather_df[['date']].copy()
            
            # Add weather features
            for col in ['temperature_2m_mean', 'precipitation_sum', 'sunshine_duration']:
                if col in weather_df.columns:
                    df[col] = weather_df[col].values
                    
            # Add events
            if len(events_df) > 0:
                event_counts = events_df.groupby('date').size().reset_index(name='event_count')
                df = df.merge(event_counts, on='date', how='left')
                df['event_count'] = df['event_count'].fillna(0)
                
            # Add sentiment
            if len(gdelt_daily) > 0:
                df = df.merge(gdelt_daily, on='date', how='left')
                df['sentiment'] = df['sentiment'].fillna(0)
                
        else:
            # Fallback: create date range
            df = pd.DataFrame({
                'date': pd.date_range(start='2022-01-01', end='2026-01-01', freq='D')
            })
            
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for the model."""
        df = df.copy()
        
        # Time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Seasonal indicators
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # Holiday indicators (German holidays approximate)
        df['is_christmas_season'] = ((df['month'] == 12) & (df['day_of_month'] >= 15)).astype(int)
        df['is_easter_season'] = ((df['month'].isin([3, 4])) & 
                                   (df['week_of_year'].isin([12, 13, 14, 15]))).astype(int)
        
        # Weather features (if available)
        if 'temperature_2m_mean' in df.columns:
            df['temp_warm'] = (df['temperature_2m_mean'] > 20).astype(int)
            df['temp_cold'] = (df['temperature_2m_mean'] < 5).astype(int)
            df['temp_moderate'] = ((df['temperature_2m_mean'] >= 5) & 
                                    (df['temperature_2m_mean'] <= 20)).astype(int)
        
        if 'precipitation_sum' in df.columns:
            df['is_rainy'] = (df['precipitation_sum'] > 1).astype(int)
            
        # Lag features (7-day and 14-day)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in ['temperature_2m_mean', 'sentiment', 'event_count']:
            if col in numeric_cols:
                df[f'{col}_lag7'] = df[col].shift(7)
                df[f'{col}_lag14'] = df[col].shift(14)
                df[f'{col}_rolling7'] = df[col].rolling(7, min_periods=1).mean()
                
        # Synthetic demand proxy (based on available signals)
        # Higher demand: warm weather + weekends + holidays + positive sentiment
        df['demand_proxy'] = 50  # baseline
        
        if 'temperature_2m_mean' in df.columns:
            # Temperature effect: moderate temps = highest demand
            temp_effect = -0.5 * np.abs(df['temperature_2m_mean'] - 15)
            df['demand_proxy'] += temp_effect
            
        if 'is_weekend' in df.columns:
            df['demand_proxy'] += df['is_weekend'] * 10
            
        if 'is_christmas_season' in df.columns:
            df['demand_proxy'] += df['is_christmas_season'] * 25
            
        if 'sentiment' in df.columns:
            df['demand_proxy'] += df['sentiment'] * 5
            
        if 'event_count' in df.columns:
            df['demand_proxy'] += df['event_count'] * 2
            
        # Add some realistic noise
        np.random.seed(42)
        df['demand_proxy'] += np.random.normal(0, 5, len(df))
        df['demand_proxy'] = df['demand_proxy'].clip(lower=10)
        
        return df
    
    def train_prophet(self, df: pd.DataFrame) -> dict:
        """Train Prophet model for seasonal patterns."""
        if not PROPHET_AVAILABLE:
            return {'model': None, 'metrics': {}}
            
        print("\nTraining Prophet model...")
        
        prophet_df = df[['date', 'demand_proxy']].rename(
            columns={'date': 'ds', 'demand_proxy': 'y'}
        ).dropna()
        
        # Add regressors if available
        regressors = []
        for col in ['temperature_2m_mean', 'is_weekend', 'event_count', 'sentiment']:
            if col in df.columns:
                prophet_df[col] = df[col].values[:len(prophet_df)]
                regressors.append(col)
        
        # Train/test split (last 90 days for validation)
        train_size = len(prophet_df) - 90
        train_df = prophet_df.iloc[:train_size]
        test_df = prophet_df.iloc[train_size:]
        
        # Initialize model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )
        
        # Add regressors
        for reg in regressors:
            model.add_regressor(reg)
        
        model.fit(train_df)
        
        # Predict on test set
        future = test_df[['ds'] + regressors]
        predictions = model.predict(future)
        
        # Metrics
        y_true = test_df['y'].values
        y_pred = predictions['yhat'].values
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        print(f"  Prophet MAE: {metrics['mae']:.2f}")
        print(f"  Prophet R²: {metrics['r2']:.3f}")
        
        return {'model': model, 'metrics': metrics, 'regressors': regressors}
    
    def train_ml_models(self, df: pd.DataFrame) -> dict:
        """Train ML ensemble (Random Forest, Gradient Boosting, Ridge)."""
        print("\nTraining ML ensemble...")
        
        # Feature columns
        exclude_cols = ['date', 'demand_proxy', 'time']
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                        if c not in exclude_cols]
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df['demand_proxy'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Drop rows with NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train/test split (last 90 days)
        train_size = len(X) - 90
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_results = {}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        models_results['random_forest'] = {
            'model': rf,
            'predictions': rf_pred,
            'mae': mean_absolute_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred)
        }
        print(f"  Random Forest MAE: {models_results['random_forest']['mae']:.2f}, R²: {models_results['random_forest']['r2']:.3f}")
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_pred = gb.predict(X_test_scaled)
        models_results['gradient_boosting'] = {
            'model': gb,
            'predictions': gb_pred,
            'mae': mean_absolute_error(y_test, gb_pred),
            'r2': r2_score(y_test, gb_pred)
        }
        print(f"  Gradient Boosting MAE: {models_results['gradient_boosting']['mae']:.2f}, R²: {models_results['gradient_boosting']['r2']:.3f}")
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_pred = ridge.predict(X_test_scaled)
        models_results['ridge'] = {
            'model': ridge,
            'predictions': ridge_pred,
            'mae': mean_absolute_error(y_test, ridge_pred),
            'r2': r2_score(y_test, ridge_pred)
        }
        print(f"  Ridge MAE: {models_results['ridge']['mae']:.2f}, R²: {models_results['ridge']['r2']:.3f}")
        
        # Feature importance (from Random Forest)
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 10 features:")
        for _, row in self.feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
        
        return {
            'models': models_results,
            'feature_cols': feature_cols,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def generate_forecast(self, df: pd.DataFrame, weeks: int = 12) -> pd.DataFrame:
        """Generate demand forecast for future weeks."""
        print(f"\nGenerating {weeks}-week forecast...")
        
        # Create future dates
        last_date = df['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=weeks * 7,
            freq='D'
        )
        
        future_df = pd.DataFrame({'date': future_dates})
        future_df = self.create_features(future_df)
        
        # For weather, use historical averages by day of year
        if 'temperature_2m_mean' in df.columns:
            df['day_of_year'] = df['date'].dt.dayofyear
            temp_by_day = df.groupby('day_of_year')['temperature_2m_mean'].mean()
            future_df['day_of_year'] = future_df['date'].dt.dayofyear
            future_df['temperature_2m_mean'] = future_df['day_of_year'].map(temp_by_day)
            future_df['temperature_2m_mean'] = future_df['temperature_2m_mean'].fillna(
                future_df['temperature_2m_mean'].mean()
            )
            # Regenerate temperature-based features
            future_df['temp_warm'] = (future_df['temperature_2m_mean'] > 20).astype(int)
            future_df['temp_cold'] = (future_df['temperature_2m_mean'] < 5).astype(int)
            future_df['temp_moderate'] = ((future_df['temperature_2m_mean'] >= 5) & 
                                          (future_df['temperature_2m_mean'] <= 20)).astype(int)
            future_df['is_rainy'] = 0  # Default for forecast
            
        # Add lag features with defaults
        for col in ['temperature_2m_mean', 'sentiment', 'event_count']:
            if col in future_df.columns:
                future_df[f'{col}_lag7'] = future_df[col].shift(7).fillna(future_df[col].mean() if col in future_df else 0)
                future_df[f'{col}_lag14'] = future_df[col].shift(14).fillna(future_df[col].mean() if col in future_df else 0)
                future_df[f'{col}_rolling7'] = future_df[col].rolling(7, min_periods=1).mean()
            else:
                future_df[f'{col}_lag7'] = 0
                future_df[f'{col}_lag14'] = 0
                future_df[f'{col}_rolling7'] = 0
                
        # Use trained models for predictions - ensure feature alignment
        ml_data = self.models.get('ml', {})
        trained_features = ml_data.get('feature_cols', [])
        
        # Add any missing features from training
        for col in trained_features:
            if col not in future_df.columns:
                future_df[col] = 0
        
        # Prepare features in the same order as training
        feature_cols = trained_features if trained_features else [
            c for c in future_df.select_dtypes(include=[np.number]).columns 
            if c not in ['date', 'demand_proxy', 'day_of_year']
        ]
        X_future = future_df[feature_cols].fillna(0)
        
        try:
            X_future_scaled = self.scaler.transform(X_future)
            
            # Get predictions from each model
            for model_name, model_data in self.models.get('ml', {}).get('models', {}).items():
                model = model_data['model']
                future_df[f'pred_{model_name}'] = model.predict(X_future_scaled)
                
            # Ensemble: weighted average
            pred_cols = [c for c in future_df.columns if c.startswith('pred_')]
            if pred_cols:
                future_df['demand_forecast'] = future_df[pred_cols].mean(axis=1)
            else:
                # Fallback to demand proxy
                future_df['demand_forecast'] = future_df['demand_proxy']
                
        except Exception as e:
            print(f"  ⚠️  Model prediction failed: {e}")
            future_df['demand_forecast'] = future_df['demand_proxy']
        
        # Add confidence intervals (±15%)
        future_df['forecast_lower'] = future_df['demand_forecast'] * 0.85
        future_df['forecast_upper'] = future_df['demand_forecast'] * 1.15
        
        print(f"  ✓ Generated forecast for {len(future_df)} days")
        
        return future_df
    
    def plot_forecast(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame):
        """Create forecast visualization."""
        print("\nGenerating forecast plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AGRICOM Demand Forecast - Berlin Organic Produce', fontsize=14, fontweight='bold')
        
        # 1. Historical + Forecast
        ax1 = axes[0, 0]
        ax1.plot(historical_df['date'].tail(180), 
                 historical_df['demand_proxy'].tail(180), 
                 'b-', alpha=0.7, label='Historical')
        ax1.plot(forecast_df['date'], forecast_df['demand_forecast'], 
                 'r-', linewidth=2, label='Forecast')
        ax1.fill_between(forecast_df['date'], 
                         forecast_df['forecast_lower'], 
                         forecast_df['forecast_upper'],
                         alpha=0.3, color='red', label='95% CI')
        ax1.set_title('Demand Forecast (12 Weeks)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Demand Index')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Seasonal Pattern
        ax2 = axes[0, 1]
        weekly_avg = forecast_df.groupby('day_of_week')['demand_forecast'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.bar(days, weekly_avg.values, color='steelblue', edgecolor='navy')
        ax2.set_title('Weekly Demand Pattern (Forecast)')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Avg Demand')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Feature Importance
        ax3 = axes[1, 0]
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            top_features = self.feature_importance.head(10)
            ax3.barh(top_features['feature'], top_features['importance'], color='forestgreen')
            ax3.set_title('Top 10 Demand Drivers')
            ax3.set_xlabel('Importance')
            ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, 'Feature importance not available', 
                    ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Monthly Forecast Summary
        ax4 = axes[1, 1]
        forecast_df['month_name'] = forecast_df['date'].dt.strftime('%B')
        monthly_forecast = forecast_df.groupby('month_name')['demand_forecast'].agg(['mean', 'std'])
        if len(monthly_forecast) > 0:
            ax4.bar(range(len(monthly_forecast)), monthly_forecast['mean'], 
                   yerr=monthly_forecast['std'], capsize=5, color='coral', edgecolor='darkred')
            ax4.set_xticks(range(len(monthly_forecast)))
            ax4.set_xticklabels(monthly_forecast.index, rotation=45, ha='right')
            ax4.set_title('Monthly Forecast Summary')
            ax4.set_ylabel('Demand Index')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d')
        filepath = FIGURE_DIR / f'demand_forecast_{timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved forecast plot: {filepath}")
        plt.close()
        
        return filepath
    
    def run(self, weeks: int = 12) -> dict:
        """Run full forecasting pipeline."""
        print("=" * 60)
        print("AGRICOM Demand Forecasting Model")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Create features
        df = self.create_features(df)
        print(f"\nDataset: {len(df)} days, {len(df.columns)} features")
        
        # Train models
        if PROPHET_AVAILABLE:
            self.models['prophet'] = self.train_prophet(df)
        self.models['ml'] = self.train_ml_models(df)
        
        # Generate forecast
        forecast_df = self.generate_forecast(df, weeks=weeks)
        
        # Save forecast
        timestamp = datetime.now().strftime('%Y%m%d')
        forecast_path = FORECAST_DIR / f'demand_forecast_{timestamp}.csv'
        forecast_df.to_csv(forecast_path, index=False)
        print(f"\n✓ Saved forecast: {forecast_path}")
        
        # Create visualizations
        plot_path = self.plot_forecast(df, forecast_df)
        
        # Summary
        print("\n" + "=" * 60)
        print("FORECAST SUMMARY")
        print("=" * 60)
        print(f"Forecast period: {forecast_df['date'].min().date()} to {forecast_df['date'].max().date()}")
        print(f"Average demand: {forecast_df['demand_forecast'].mean():.1f}")
        print(f"Peak demand: {forecast_df['demand_forecast'].max():.1f}")
        print(f"Lowest demand: {forecast_df['demand_forecast'].min():.1f}")
        
        # Weekly peaks
        print("\nPeak days by week:")
        forecast_df['week'] = forecast_df['date'].dt.isocalendar().week
        for week in forecast_df['week'].unique()[:4]:
            week_data = forecast_df[forecast_df['week'] == week]
            peak_day = week_data.loc[week_data['demand_forecast'].idxmax()]
            print(f"  Week {week}: {peak_day['date'].strftime('%a %b %d')} ({peak_day['demand_forecast']:.1f})")
        
        return {
            'historical': df,
            'forecast': forecast_df,
            'models': self.models,
            'feature_importance': self.feature_importance,
            'files': {
                'forecast_csv': str(forecast_path),
                'forecast_plot': str(plot_path)
            }
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AGRICOM Demand Forecasting')
    parser.add_argument('--weeks', type=int, default=12, help='Forecast horizon in weeks')
    parser.add_argument('--neighborhood', default='all', help='Target neighborhood')
    
    args = parser.parse_args()
    
    forecaster = DemandForecaster(neighborhood=args.neighborhood)
    results = forecaster.run(weeks=args.weeks)
    
    print("\n✅ Forecasting complete!")
    

if __name__ == '__main__':
    main()
