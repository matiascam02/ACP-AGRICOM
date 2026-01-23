"""
AGRICOM - Weather Data Collection
Collects historical and forecast weather data for Berlin using Open-Meteo API.

Usage:
    python weather.py

Output:
    data/raw/weather_berlin_YYYYMMDD.csv
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')

# Berlin coordinates
BERLIN_LAT = 52.52
BERLIN_LON = 13.405

# Open-Meteo API (free, no API key required)
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Weather variables to collect
DAILY_VARIABLES = [
    'temperature_2m_max',
    'temperature_2m_min',
    'temperature_2m_mean',
    'precipitation_sum',
    'rain_sum',
    'snowfall_sum',
    'precipitation_hours',
    'wind_speed_10m_max',
    'wind_gusts_10m_max',
    'sunshine_duration',
    'et0_fao_evapotranspiration',  # Reference evapotranspiration
]


def fetch_historical_weather(start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo Archive API.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        DataFrame with daily weather data
    """
    params = {
        'latitude': BERLIN_LAT,
        'longitude': BERLIN_LON,
        'start_date': start_date,
        'end_date': end_date,
        'daily': ','.join(DAILY_VARIABLES),
        'timezone': 'Europe/Berlin'
    }

    print(f"Fetching historical data: {start_date} to {end_date}")

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()

    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    return df


def fetch_forecast_weather(days=14):
    """
    Fetch weather forecast from Open-Meteo Forecast API.

    Args:
        days: Number of days to forecast

    Returns:
        DataFrame with daily forecast data
    """
    params = {
        'latitude': BERLIN_LAT,
        'longitude': BERLIN_LON,
        'daily': ','.join(DAILY_VARIABLES),
        'timezone': 'Europe/Berlin',
        'forecast_days': days
    }

    print(f"Fetching {days}-day forecast")

    response = requests.get(FORECAST_URL, params=params)
    response.raise_for_status()

    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data['daily'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df['is_forecast'] = True

    return df


def calculate_anomalies(df, reference_years=5):
    """
    Calculate temperature anomalies (deviation from historical average for same day-of-year).

    Args:
        df: DataFrame with weather data
        reference_years: Years to use for baseline calculation

    Returns:
        DataFrame with anomaly columns added
    """
    df = df.copy()

    # Calculate day of year
    df['day_of_year'] = df.index.dayofyear

    # Calculate historical averages by day of year
    # (Using available data as reference)
    daily_means = df.groupby('day_of_year').agg({
        'temperature_2m_mean': 'mean',
        'precipitation_sum': 'mean'
    }).rename(columns={
        'temperature_2m_mean': 'temp_historical_mean',
        'precipitation_sum': 'precip_historical_mean'
    })

    # Merge back
    df = df.merge(daily_means, left_on='day_of_year', right_index=True, how='left')

    # Calculate anomalies
    df['temp_anomaly'] = df['temperature_2m_mean'] - df['temp_historical_mean']
    df['precip_anomaly'] = df['precipitation_sum'] - df['precip_historical_mean']

    # Categorize anomalies
    df['temp_anomaly_category'] = pd.cut(
        df['temp_anomaly'],
        bins=[-float('inf'), -5, -2, 2, 5, float('inf')],
        labels=['very_cold', 'cold', 'normal', 'warm', 'very_warm']
    )

    return df


def add_derived_features(df):
    """
    Add derived weather features useful for demand forecasting.

    Args:
        df: DataFrame with weather data

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Temperature range
    df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']

    # Binary indicators
    df['is_rainy'] = df['precipitation_sum'] > 1.0  # More than 1mm
    df['is_hot'] = df['temperature_2m_max'] > 25  # Above 25°C
    df['is_cold'] = df['temperature_2m_min'] < 5   # Below 5°C
    df['is_windy'] = df['wind_speed_10m_max'] > 40  # Above 40 km/h

    # Season
    df['month'] = df.index.month
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })

    # Day of week
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    return df


def save_data(df, filename_prefix='weather_berlin'):
    """Save data to CSV with timestamp."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{filename_prefix}_{timestamp}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(filepath)
    print(f"\nSaved to: {filepath}")
    return filepath


def main():
    """Main collection routine."""
    print("=" * 60)
    print("AGRICOM - Weather Data Collection")
    print("=" * 60)
    print(f"Location: Berlin ({BERLIN_LAT}, {BERLIN_LON})")
    print("=" * 60)

    # Calculate date range (last 3 years)
    end_date = datetime.now() - timedelta(days=7)  # Archive has ~1 week delay
    start_date = end_date - timedelta(days=3*365)

    # Fetch historical data
    print("\n1. Fetching historical weather data...")
    df_historical = fetch_historical_weather(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    df_historical['is_forecast'] = False

    # Fetch forecast
    print("\n2. Fetching weather forecast...")
    try:
        df_forecast = fetch_forecast_weather(days=14)

        # Combine
        df = pd.concat([df_historical, df_forecast])
    except Exception as e:
        print(f"Warning: Could not fetch forecast: {e}")
        df = df_historical

    # Calculate anomalies
    print("\n3. Calculating temperature anomalies...")
    df = calculate_anomalies(df)

    # Add derived features
    print("\n4. Adding derived features...")
    df = add_derived_features(df)

    # Save
    print("\n5. Saving data...")
    filepath = save_data(df)

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total days: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nTemperature stats:")
    print(f"  Mean: {df['temperature_2m_mean'].mean():.1f}°C")
    print(f"  Min:  {df['temperature_2m_min'].min():.1f}°C")
    print(f"  Max:  {df['temperature_2m_max'].max():.1f}°C")
    print("\nPrecipitation stats:")
    print(f"  Total rainy days: {df['is_rainy'].sum()}")
    print(f"  Avg daily precip: {df['precipitation_sum'].mean():.1f}mm")
    print("\nAnomaly distribution:")
    print(df['temp_anomaly_category'].value_counts())


if __name__ == "__main__":
    main()
