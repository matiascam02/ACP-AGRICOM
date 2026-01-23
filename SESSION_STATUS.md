# AGRICOM Project - Session Status
**Last Updated:** 2026-01-22 20:30
**Read this first in new sessions**

---

## Project Summary
- **Client:** AgriCom (organic produce company)
- **Goal:** Predict organic produce demand in Berlin using alternative data signals
- **Team:** 6 students + mentor Ryan Krog (ESMT)

---

## Data Collected (Complete)

| Source | File | Records |
|--------|------|---------|
| Weather | `data/raw/weather_berlin_20260122.csv` | 1,110 days |
| Events | `data/raw/events_berlin_20260122.csv` | 196 events |
| GDELT Articles | `data/raw/gdelt_articles_20260122.csv` | 159 articles |
| GDELT Timeline | `data/raw/gdelt_timeline_20260122.csv` | 2,856 sentiment points |
| Neighborhood Profiles | `outputs/reports/neighborhood_profiles.csv` | 3 segments |

## Data Pending (Needs Team)

| Source | Status | Action |
|--------|--------|--------|
| Google Trends | Rate limited | **Team must download manually** |
| Reddit | Script ready | Needs API credentials |

---

## Scripts Created

### Data Collection (`src/data_collection/`)
- `weather.py` - Open-Meteo API (working)
- `events.py` - Bundesliga + holidays (working)
- `gdelt_news.py` - News sentiment (working)
- `google_trends_chunked.py` - Rate limited, use manual download
- `reddit_collector.py` - Ready, needs API credentials

### Analysis (`src/analysis/`)
- `weather_analysis.py` - Weather visualizations
- `trends_weather_analysis_v2.py` - Trends + weather correlation
- `gdelt_analysis.py` - News sentiment analysis
- `neighborhood_segmentation.py` - Berlin segment profiles

---

## Visualizations Generated (`outputs/figures/`)

1. `weather_temperature_overview.png`
2. `weather_seasonal_patterns.png`
3. `weather_demand_hypothesis.png`
4. `trends_timeseries.png`
5. `trends_weather_correlation.png`
6. `trends_monthly_patterns.png`
7. `gdelt_article_coverage.png`
8. `gdelt_sentiment_analysis.png`
9. `gdelt_source_analysis.png`
10. `neighborhood_segmentation.png`
11. `neighborhood_demand_drivers.png`

---

## Obsidian Documentation

Location: `40_Projects/ESMT/AGRICOM/AGRICOM/claudes-work/`

Key files:
- `00_Index.md` - Work log and navigation
- `07_Visualizations.md` - All charts with insights
- `08_Additional_Data_Sources.md` - GDELT results
- `09_Team_Tasks_and_Next_Steps.md` - **Team keyword assignments**

---

## Immediate Next Steps

### 1. Team Action (Manual)
Each team member downloads 2-3 Google Trends keywords:
- Go to trends.google.com
- Search keyword → Germany → 5 years
- Download CSV → save to `data/raw/google_trends/`

**Keywords assigned in:** `claudes-work/09_Team_Tasks_and_Next_Steps.md`

### 2. After Team Uploads Trends
```bash
cd /Users/matiascam/Documents/2_Education/ESMT/AGRICOM/agricom_project
source venv/bin/activate
python src/analysis/trends_weather_analysis_v2.py
```

### 3. Build Forecast Model
- Merge all data sources
- Create lag features
- Train Prophet/ARIMA model
- Generate neighborhood-specific forecasts

---

## Key Findings So Far

1. **GDELT Sentiment:** Organic food coverage is positive (+0.32), "Bio-Lebensmittel sind so gefragt wie nie"
2. **Weather:** 47% warm days (salads), 34% rainy days (convenience)
3. **Trends:** Christmas spikes 2-3x, weak temp correlation (r=-0.12)
4. **Segments:** Kreuzberg = primary target (85% organic affinity)

---

## Virtual Environment

```bash
cd /Users/matiascam/Documents/2_Education/ESMT/AGRICOM/agricom_project
source venv/bin/activate
```

Installed: pandas, numpy, matplotlib, seaborn, scipy, pytrends, gdeltdoc, praw

---

## Questions for Next Session

1. Has team uploaded Google Trends data?
2. Reddit API credentials available?
3. Ready to build forecast model?
4. Need presentation materials?
