# AGRICOM Project - Session Status
**Last Updated:** 2026-01-27 14:10
**Read this first in new sessions**

---

## Project Summary
- **Client:** AgriCom (organic produce company)
- **Goal:** Predict organic produce demand in Berlin using alternative data signals
- **Team:** 6 students + mentor Ryan Krog (ESMT)
- **Timeline:** January - March 2026

---

## Data Collected (Complete)

| Source | File | Records |
|--------|------|---------|
| Weather | `data/raw/weather_berlin_20260122.csv` | 1,110 days |
| Events | `data/raw/events_berlin_20260122.csv` | 196 events |
| GDELT Articles | `data/raw/gdelt_articles_20260122.csv` | 159 articles |
| GDELT Timeline | `data/raw/gdelt_timeline_20260122.csv` | 2,856 sentiment points |
| Neighborhood Profiles | `outputs/reports/neighborhood_profiles.csv` | 3 segments |

## Data Partial (Needs Team)

| Source | Status | Action |
|--------|--------|--------|
| Google Trends | 8/20 keywords | **Team must download manually** |

See `data/raw/google_trends/DOWNLOAD_GUIDE.md` for instructions.

---

## New Scripts Added (2026-01-27) 🆕

### Data Collection
| Script | Purpose | Requirements |
|--------|---------|--------------|
| `google_trends_serpapi.py` | Alternative trends collection | SerpAPI key (optional) |
| `economic_indicators.py` | OECD, Eurostat, World Bank data | None (APIs are free) |
| `social_signals.py` | YouTube, Reddit, Instagram | API keys (optional) |

### Analysis
| Script | Purpose |
|--------|---------|
| `data_merger.py` | Combines ALL sources into unified dataset |

---

## Scripts Overview

### Data Collection (`src/data_collection/`)
- `weather.py` - Open-Meteo API ✅
- `events.py` - Bundesliga + holidays ✅
- `gdelt_news.py` - News sentiment ✅
- `google_trends_chunked.py` - Rate limited, use manual download
- `google_trends_serpapi.py` - Alternative with SerpAPI 🆕
- `economic_indicators.py` - Economic data 🆕
- `social_signals.py` - Social media signals 🆕
- `reddit_collector.py` - Ready, needs API credentials

### Analysis (`src/analysis/`)
- `weather_analysis.py` - Weather visualizations ✅
- `trends_weather_analysis_v2.py` - Trends + weather correlation ✅
- `gdelt_analysis.py` - News sentiment analysis ✅
- `neighborhood_segmentation.py` - Berlin segment profiles ✅
- `data_merger.py` - Unified dataset creation 🆕

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

---

## Key Findings So Far

1. **GDELT Sentiment:** Organic food coverage is positive (+0.32)
2. **Weather:** 47% warm days (salads), 34% rainy days (convenience)
3. **Trends:** Christmas spikes 2-3x, weak temp correlation (r=-0.12)
4. **Segments:** Kreuzberg = primary target (85% organic affinity)

---

## Immediate Next Steps

### 1. Team Action: Download Google Trends ⏰
See `data/raw/google_trends/DOWNLOAD_GUIDE.md`

**Deadline:** Before next team meeting

### 2. After Team Uploads Trends
```bash
cd ACP-AGRICOM
pip install -r requirements.txt
python src/analysis/data_merger.py
```

### 3. Build Forecast Model
- Merge all data sources
- Create lag features
- Train Prophet/ARIMA model
- Generate neighborhood-specific forecasts

---

## Meeting Preparation

See `MEETING_PREP.md` for:
- Team meeting agenda
- Action items by person
- Questions for mentor/AgriCom
- Project timeline

---

## Virtual Environment Setup

```bash
cd ACP-AGRICOM
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Quick Commands

```bash
# Check trends collection status
python src/data_collection/google_trends_serpapi.py --status

# Generate download guide
python src/data_collection/google_trends_serpapi.py --guide

# Collect economic data
python src/data_collection/economic_indicators.py

# Merge all data (after trends uploaded)
python src/analysis/data_merger.py

# Run weather analysis
python src/analysis/weather_analysis.py

# Run GDELT analysis
python src/analysis/gdelt_analysis.py
```

---

## Questions for Next Session

1. Has team uploaded Google Trends data?
2. Reddit API credentials available?
3. Ready to build forecast model?
4. Need presentation materials for midpoint?

---

## Repository

- **GitHub:** https://github.com/matiascam02/ACP-AGRICOM
- **Obsidian Docs:** `40_Projects/ESMT/AGRICOM/AGRICOM/claudes-work/`
