# Preprocessing and Feature Engineering

This document describes all data transformations, feature engineering steps, and data alignment procedures applied before feeding data into the forecasting model.

---

## 1. Weather Data Preprocessing

**Script:** `src/data_collection/weather.py` (applied at collection time)

### Temperature Anomaly Calculation

Historical temperature data is used to compute daily temperature anomalies -- the deviation of each day's mean temperature from the historical average for that calendar day.

1. Compute `day_of_year` from the date index.
2. Group all available data by `day_of_year` and calculate the historical mean for `temperature_2m_mean` and `precipitation_sum`.
3. Compute anomalies:
   - `temp_anomaly` = `temperature_2m_mean` - `temp_historical_mean`
   - `precip_anomaly` = `precipitation_sum` - `precip_historical_mean`
4. Categorize temperature anomalies into bins:

| Category | Anomaly Range |
|---|---|
| `very_cold` | < -5°C |
| `cold` | -5°C to -2°C |
| `normal` | -2°C to +2°C |
| `warm` | +2°C to +5°C |
| `very_warm` | > +5°C |

### Derived Weather Features

The following binary and categorical features are derived from the raw weather observations:

| Feature | Logic | Purpose |
|---|---|---|
| `temp_range` | `temperature_2m_max` - `temperature_2m_min` | Captures daily temperature variability |
| `is_rainy` | `precipitation_sum` > 1.0 mm | Binary flag for rain days |
| `is_hot` | `temperature_2m_max` > 25°C | Binary flag for hot days |
| `is_cold` | `temperature_2m_min` < 5°C | Binary flag for cold days |
| `is_windy` | `wind_speed_10m_max` > 40 km/h | Binary flag for windy days |
| `month` | Extracted from date | Calendar month (1--12) |
| `season` | Mapped from month | `winter` (Dec--Feb), `spring` (Mar--May), `summer` (Jun--Aug), `autumn` (Sep--Nov) |
| `day_of_week` | Extracted from date | Day of week (0=Monday, 6=Sunday) |
| `is_weekend` | `day_of_week` in [5, 6] | Binary flag for weekends |

---

## 2. Events Data Preprocessing

**Script:** `src/analysis/data_merger.py` (`load_events_data`)

Raw event records are aggregated to a daily summary suitable for merging:

1. **Group by date** and count the number of events per day (`event_count`).
2. **Concatenate event types** per day into a comma-separated string.
3. **Create binary flags:**
   - `has_bundesliga`: True if any event on that date contains "Bundesliga" in the event type.
   - `has_holiday`: True if any event on that date contains "holiday" (case-insensitive) in the event type.

When aggregating to weekly frequency, event counts and binary flags are summed (i.e., the weekly value represents the total number of event days or flagged days within that week).

---

## 3. GDELT Sentiment Preprocessing

**Script:** `src/analysis/data_merger.py` (`load_gdelt_data`)

The GDELT timeline data arrives at sub-daily granularity and is aggregated to daily:

1. Parse the `datetime` column and extract the date portion.
2. Group by date and compute:
   - `gdelt_sentiment_mean`: Daily mean of `Average Tone`
   - `gdelt_sentiment_std`: Daily standard deviation of `Average Tone`
   - `gdelt_article_count`: Number of sentiment data points per day

When aggregating to weekly frequency, sentiment mean is averaged and article count is summed.

---

## 4. Google Trends Preprocessing

**Scripts:** `src/analysis/data_merger.py` (`load_trends_data`), `src/data_collection/google_trends_serpapi.py` (`import_manual_csvs`, `combine_all_trends`)

### CSV Parsing

Google Trends CSV files exported from the web interface contain header metadata rows. The loading process:

1. Skip the first 1--2 rows (category/header metadata).
2. Rename the first column to `date` and the second to `value`.
3. Parse dates and drop rows with invalid dates.
4. Extract the keyword name from the filename by stripping the `trends_` prefix and `_de_5y` suffix, and replacing underscores with spaces.

### Pivot to Wide Format

Individual keyword time series are combined into a single wide-format DataFrame:

1. Concatenate all keyword DataFrames (long format: `date`, `value`, `keyword`).
2. Pivot using `pivot_table` with `date` as the index, `keyword` as columns, and `value` as values (aggregated by mean if duplicates exist).
3. Rename columns with a `trends_` prefix (e.g., `trends_bio_lebensmittel`, `trends_wochenmarkt_berlin`).

### Composite Trends Index

A `trends_composite` column is created as the arithmetic mean of all `trends_*` columns for each row. This serves as a single aggregate measure of organic food search interest.

### Weekly Standardization

In `google_trends_serpapi.py`, dates are standardized to weekly periods using `dt.to_period('W').dt.start_time` to align with Google Trends' native weekly granularity.

---

## 5. Economic Indicators Preprocessing

**Script:** `src/data_collection/economic_indicators.py`

### Source Combination

Data from multiple APIs (OECD, World Bank, Eurostat, Bundesbank) arrives at different frequencies and date ranges. The combination process:

1. Collect all dates from all sources to determine the overall date range.
2. Create a master monthly date range (`freq='MS'`, month-start).
3. Resample each source to monthly frequency by grouping on month period and taking the first value.
4. Merge all sources onto the master date range via left join on date.
5. Forward fill missing values across the combined DataFrame.

### Derived Economic Features

| Feature | Logic | Purpose |
|---|---|---|
| `*_mom_change` | `pct_change() * 100` on each numeric column | Month-over-month percentage change |
| `economic_sentiment` | Categorical from `consumer_confidence`: >100 = "optimistic", <98 = "pessimistic", else "neutral" | Simplified sentiment signal |
| `inflation_category` | Binned from `inflation_rate`: <2 = "low", 2--4 = "moderate", 4--6 = "high", >6 = "very_high" | Inflation regime indicator |

---

## 6. Data Alignment and Merging

**Script:** `src/analysis/data_merger.py` (`create_unified_dataset`)

All data sources operate at different native frequencies:

| Source | Native Frequency |
|---|---|
| Weather | Daily |
| Events | Event-based (irregular) |
| GDELT Sentiment | Sub-daily |
| Google Trends | Weekly |
| Economic Indicators | Monthly |
| Social Signals | Point-in-time snapshot |

The data merger aligns all sources to a common frequency (daily or weekly) using the following strategy:

### Base Date Range

A continuous date range is created spanning from the earliest date to the latest date found across all data sources.

### Merge Strategy by Source

| Source | Daily Alignment | Weekly Alignment |
|---|---|---|
| **Weather** | Direct left join on date | Resample: mean for temperature, sum for precipitation, sum for binary flags (rainy/hot/cold day counts) |
| **Events** | Direct left join on date (already aggregated daily) | Resample: sum event_count, sum has_bundesliga, sum has_holiday |
| **GDELT** | Direct left join on date (already aggregated daily) | Resample: mean for sentiment, sum for article count |
| **Google Trends** | Forward fill weekly values to daily, then left join | Resample: mean of weekly values, then left join |
| **Economic** | Forward fill monthly values to daily, then left join | Forward fill monthly values to weekly, then left join |

### Handling Missing Values

After all merges, remaining missing values are filled using a two-pass strategy:

1. **Forward fill** (`ffill`): Propagate the last known value forward in time.
2. **Backward fill** (`bfill`): Fill any remaining leading NaNs with the next known value.

---

## 7. Feature Engineering for the Forecasting Model

**Script:** `src/analysis/demand_forecast.py` (`DemandForecaster.create_features`)

The forecasting model applies a comprehensive set of engineered features on top of the merged data.

### Time-Based Features

| Feature | Derivation |
|---|---|
| `day_of_week` | 0 (Monday) to 6 (Sunday) from the date |
| `day_of_month` | 1--31 from the date |
| `week_of_year` | ISO week number (1--53) |
| `month` | Calendar month (1--12) |
| `quarter` | Calendar quarter (1--4) |
| `year` | Calendar year |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |

### Seasonal Indicator Features

| Feature | Logic |
|---|---|
| `is_winter` | 1 if month in {12, 1, 2} |
| `is_spring` | 1 if month in {3, 4, 5} |
| `is_summer` | 1 if month in {6, 7, 8} |
| `is_fall` | 1 if month in {9, 10, 11} |

### Holiday Indicator Features

| Feature | Logic |
|---|---|
| `is_christmas_season` | 1 if month = 12 and day >= 15 |
| `is_easter_season` | 1 if month in {3, 4} and ISO week in {12, 13, 14, 15} |

### Weather-Derived Model Features

| Feature | Logic |
|---|---|
| `temp_warm` | 1 if `temperature_2m_mean` > 20°C |
| `temp_cold` | 1 if `temperature_2m_mean` < 5°C |
| `temp_moderate` | 1 if `temperature_2m_mean` between 5°C and 20°C |
| `is_rainy` | 1 if `precipitation_sum` > 1 mm |

### Lag Features

Lag features capture the state of key signals at previous time steps:

| Base Variable | Lags Created | Purpose |
|---|---|---|
| `temperature_2m_mean` | 7-day, 14-day | Capture delayed weather effects on demand |
| `sentiment` | 7-day, 14-day | Capture delayed media sentiment effects |
| `event_count` | 7-day, 14-day | Capture delayed event effects |

### Rolling Window Features

Rolling averages smooth out short-term noise:

| Base Variable | Windows | Aggregation |
|---|---|---|
| `temperature_2m_mean` | 7-day | Mean |
| `sentiment` | 7-day | Mean |
| `event_count` | 7-day | Mean |

### Additional Lag and Rolling Features (Data Merger)

The data merger adds further lag and rolling features to the unified dataset before the model sees it:

| Feature Type | Lags / Windows | Aggregations |
|---|---|---|
| Lag features | 1, 2, 4 weeks | Shifted values |
| Rolling features | 4, 8 weeks | Rolling mean, rolling standard deviation |

These are applied to the top key columns containing `temp`, `precip`, `trends`, or `sentiment` in their names.

---

## 8. Demand Proxy (Target Variable)

**Script:** `src/analysis/demand_forecast.py` (`DemandForecaster.create_features`)

Since the project uses alternative data to predict organic produce demand and no direct sales data is available, a synthetic demand proxy is constructed as the target variable for model training. The proxy is built from the available signals:

| Component | Formula | Weight |
|---|---|---|
| Baseline | Constant value | 50 |
| Temperature effect | -0.5 * abs(`temperature_2m_mean` - 15) | Peaks at 15°C, drops for extreme temps |
| Weekend bonus | +10 if weekend | +10 |
| Christmas season bonus | +25 if Christmas season | +25 |
| Sentiment effect | `sentiment` * 5 | Positive sentiment increases demand |
| Event effect | `event_count` * 2 | More events increase demand |
| Noise | Normal(0, 5), seed=42 | Gaussian noise for realism |

The final `demand_proxy` is clipped to a minimum of 10.

---

## 9. Forecast-Time Feature Construction

**Script:** `src/analysis/demand_forecast.py` (`DemandForecaster.generate_forecast`)

When generating forecasts for future dates (beyond the training period), the feature construction adapts to the absence of actuals:

1. **Temperature:** Historical averages by day-of-year are used as stand-ins for future temperature values. Temperature-derived features (`temp_warm`, `temp_cold`, `temp_moderate`) are re-derived from these averages.
2. **Precipitation:** Set to 0 (no rain assumed) for the forecast horizon.
3. **Lag features:** Shifted from the generated future features themselves; initial values filled with column means where history is unavailable.
4. **Rolling features:** Computed over the future feature DataFrame with `min_periods=1`.
5. **Missing model features:** Any feature present during training but absent from the future DataFrame is filled with 0.

---

## 10. Correlation and Alignment for Analysis

**Scripts:** `src/analysis/correlation_analysis.py`, `src/analysis/trends_weather_analysis.py`, `src/analysis/trends_weather_analysis_v2.py`

### Weather-to-Trends Alignment

Weather data (daily) is resampled to weekly (ending Sunday, `W-SUN`) to match Google Trends' native weekly frequency:

| Weather Variable | Weekly Aggregation |
|---|---|
| `temperature_2m_mean` | Mean |
| `temperature_2m_max` | Max |
| `temperature_2m_min` | Min |
| `temp_anomaly` | Mean |
| `precipitation_sum` | Sum |
| `is_rainy` | Sum (count of rainy days in the week) |

The resampled weather is then joined with the trends DataFrame on matching week-start dates using an inner join.

### Cross-Correlation Analysis

Cross-correlations between weather variables and trend keywords are computed at multiple lags (up to +/-12 weeks):

1. Align both series via concatenation and drop NaN rows.
2. Standardize both series (subtract mean, divide by standard deviation).
3. Compute Pearson correlation at each integer lag from -12 to +12 weeks.

### Granger Causality Testing

The `statsmodels` library's `grangercausalitytests` function is used with a maximum lag of 4 periods to test whether one time series Granger-causes another. The minimum p-value across all tested lags determines significance at the 0.05 level.

### Zero-Value Handling for Organic Terms

For Google Trends keywords containing "bio" in their name, zero values are treated as missing data (not actual zero search interest) and are excluded before computing correlations.

---

## 11. Model Training Pipeline

**Script:** `src/analysis/demand_forecast.py` (`DemandForecaster.train_ml_models`)

### Feature Selection

All numeric columns in the dataset are used as features, excluding `date`, `demand_proxy` (the target), and `time`.

### Missing Value Handling

Feature columns are filled with column-wise medians before training.

### Scaling

Features are standardized using `sklearn.preprocessing.StandardScaler` (zero mean, unit variance), fit on the training set and applied to both training and test sets.

### Train/Test Split

A time-based split is used: the last 90 days of data form the test set, and all preceding data forms the training set. No shuffling is applied (respecting temporal ordering). Cross-validation uses `TimeSeriesSplit` with 5 folds.

### Models Trained

| Model | Hyperparameters |
|---|---|
| Random Forest | 100 estimators, max depth 10, random state 42 |
| Gradient Boosting | 100 estimators, max depth 5, random state 42 |
| Ridge Regression | alpha = 1.0 |

### Ensemble Prediction

The final demand forecast is the arithmetic mean of predictions from all three models. Confidence intervals are computed as +/-15% of the ensemble prediction.

### Feature Importance

Feature importance is extracted from the Random Forest model's `feature_importances_` attribute and stored for reporting.
