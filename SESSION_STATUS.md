# AGRICOM Project - Session Status
**Last Updated:** 2026-01-27 14:55
**Read this first in new sessions**

---

## Project Summary
- **Client:** AgriCom (organic produce company)
- **Goal:** Predict organic produce demand in Berlin using alternative data signals
- **Team:** 6 students + mentor Ryan Krog (ESMT)
- **Timeline:** January - March 2026

---

## ✅ LATEST: Forecast Model Complete! (2026-01-27)

**New deliverables:**
- `src/analysis/demand_forecast.py` - Full forecasting pipeline
- `outputs/forecasts/demand_forecast_20260127.csv` - 12-week forecast
- `outputs/figures/demand_forecast_20260127.png` - Forecast visualization
- `data/raw/economic_indicators_20260127.csv` - Economic data (inflation, GDP, food prices)

**Model Performance (on 90-day holdout):**
- Ridge Regression: MAE 3.96, R² 0.818 ⭐ Best
- Random Forest: MAE 4.76, R² 0.772
- Gradient Boosting: MAE 4.84, R² 0.729

**Top Demand Drivers:**
1. Day of week (24.1%) - Weekends peak
2. Christmas season (21.5%) - 2-3x spike
3. Sentiment (13.1%) - News tone matters
4. Temperature (8.8%) - Moderate = best
5. Weekend flag (7.7%)

---

## Data Collected (Complete)

| Source | File | Records |
|--------|------|---------|
| Weather | `data/raw/weather_berlin_20260122.csv` | 1,110 days |
| Events | `data/raw/events_berlin_20260122.csv` | 196 events |
| GDELT Articles | `data/raw/gdelt_articles_20260122.csv` | 159 articles |
| GDELT Timeline | `data/raw/gdelt_timeline_20260122.csv` | 2,856 sentiment points |
| Neighborhood Profiles | `outputs/reports/neighborhood_profiles.csv` | 3 segments |
| Economic Indicators | `data/raw/economic_indicators_20260127.csv` | 360 months 🆕 |

## Data Partial (Needs Team)

| Source | Status | Action |
|--------|--------|--------|
| Google Trends | 8/20 keywords | **Team must download manually** |

See `data/raw/google_trends/DOWNLOAD_GUIDE.md` for instructions.

---

## Scripts Overview

### Data Collection (`src/data_collection/`)
- `weather.py` - Open-Meteo API ✅
- `events.py` - Bundesliga + holidays ✅
- `gdelt_news.py` - News sentiment ✅
- `economic_indicators.py` - OECD, Eurostat, World Bank ✅ 🆕
- `google_trends_serpapi.py` - Alternative with SerpAPI
- `social_signals.py` - YouTube, Reddit, Instagram
- `reddit_collector.py` - Ready, needs API credentials

### Analysis (`src/analysis/`)
- `weather_analysis.py` - Weather visualizations ✅
- `trends_weather_analysis_v2.py` - Trends + weather correlation ✅
- `gdelt_analysis.py` - News sentiment analysis ✅
- `neighborhood_segmentation.py` - Berlin segment profiles ✅
- `data_merger.py` - Unified dataset creation
- `demand_forecast.py` - **ML forecasting pipeline** ✅ 🆕

---

## Visualizations Generated (`outputs/figures/`)

1. `weather_temperature_overview.png`
2. `weather_seasonal_patterns.png`
3. `weather_demand_hypothesis.png`
4. `trends_timeseries.png`
5. `trends_weather_correlation.png`
6. `trends_monthly_patterns.png`
7. `trends_seasonal_patterns.png`
8. `trends_supermarkt_analysis.png`
9. `gdelt_article_coverage.png`
10. `gdelt_sentiment_analysis.png`
11. `gdelt_source_analysis.png`
12. `neighborhood_segmentation.png`
13. `neighborhood_demand_drivers.png`
14. `demand_forecast_20260127.png` 🆕

---

## Key Findings

1. **GDELT Sentiment:** Organic food coverage is positive (+0.32)
2. **Weather:** 47% warm days (salads), 34% rainy days (convenience)
3. **Trends:** Christmas spikes 2-3x, weak temp correlation (r=-0.12)
4. **Segments:** Kreuzberg = primary target (85% organic affinity)
5. **Forecast:** Weekends + Christmas = highest demand; temperature moderates effect 🆕

---

## Forecast Summary (Feb-Apr 2026)

```
Forecast period: 2026-02-05 to 2026-04-29
Average demand index: 49.4
Peak demand: 59.2
Lowest demand: 42.5

Peak days trend toward WEEKENDS (Sat/Sun)
```

---

## Next Steps

### 1. Team Action: Download Google Trends ⏰
See `data/raw/google_trends/DOWNLOAD_GUIDE.md`

### 2. Run Full Pipeline (After Trends)
```bash
python src/analysis/data_merger.py      # Merge all data
python src/analysis/demand_forecast.py  # Generate forecast
```

### 3. Prepare Presentation
- Create slides for midpoint review
- Include forecast charts
- Highlight neighborhood recommendations

---

## Quick Commands

```bash
# Run demand forecast
python src/analysis/demand_forecast.py --weeks 12

# Collect economic data
python src/data_collection/economic_indicators.py

# Merge all data
python src/analysis/data_merger.py

# Run weather analysis
python src/analysis/weather_analysis.py

# Run GDELT analysis
python src/analysis/gdelt_analysis.py
```

---

## Repository

- **GitHub:** https://github.com/matiascam02/ACP-AGRICOM
- **Obsidian Docs:** `40_Projects/ESMT/AGRICOM/AGRICOM/claudes-work/`
