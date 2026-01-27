# AGRICOM Team Meeting Preparation
**Generated:** 2026-01-27
**Meeting Focus:** Data Collection Status & Next Steps

---

## 📊 Current Progress Summary

### Data Sources Collected ✅

| Source | Status | Records | Quality |
|--------|--------|---------|---------|
| Weather (Berlin) | ✅ Complete | 1,110 days | High |
| Events (Berlin) | ✅ Complete | 196 events | High |
| GDELT News Sentiment | ✅ Complete | 159 articles | Medium |
| GDELT Timeline | ✅ Complete | 2,856 points | High |
| Neighborhood Segmentation | ✅ Complete | 3 profiles | High |

### Data Sources Partial ⚠️

| Source | Status | Collected | Missing |
|--------|--------|-----------|---------|
| Google Trends | ⚠️ Partial | 8 keywords | ~12 keywords |

### Data Sources Pending 🔄

| Source | Status | Blocker |
|--------|--------|---------|
| Reddit | Ready | Needs API credentials |
| Economic Indicators | New script | Ready to run |
| Social Signals | New script | Needs API keys (optional) |

---

## 🎯 Key Findings So Far

### 1. News Sentiment (GDELT)
- **Average sentiment:** +0.32 (positive)
- Key insight: "Bio-Lebensmittel sind so gefragt wie nie" (Organic food more popular than ever)
- Coverage peaks around health/sustainability news cycles

### 2. Weather Patterns
- 47% of days are "warm" (good for salads, fresh produce)
- 34% of days have rain (drives convenience/delivery demand)
- Clear seasonal patterns identified

### 3. Google Trends (Preliminary)
- Christmas spikes: 2-3x normal interest
- Weak temperature-trends correlation (r = -0.12)
- "Bio Supermarkt" searches peak on weekends

### 4. Neighborhood Segmentation
| Segment | Organic Affinity | Key Driver |
|---------|------------------|------------|
| Kreuzberg | 85% | Farmers markets, sustainability |
| Mitte | 70% | Convenience, premium quality |
| Charlottenburg | 60% | Family health, weekend shopping |

---

## 🚨 Action Items for Team

### URGENT: Google Trends Download (5 min per person)

Each team member should download 2-3 keywords manually:

#### Team Member 1
- [ ] `bio lebensmittel` (Germany)
- [ ] `wochenmarkt berlin` (Berlin)
- [ ] `alnatura` (Germany)

#### Team Member 2
- [ ] `bio gemüse` (Germany)
- [ ] `bauernmarkt berlin` (Berlin)
- [ ] `rewe bio` (Germany)

#### Team Member 3
- [ ] `bio obst` (Germany)
- [ ] `bio supermarkt` (Germany)
- [ ] `edeka bio` (Germany)

#### Team Member 4
- [ ] `nachhaltig einkaufen` (Germany)
- [ ] `vegan lebensmittel` (Germany)
- [ ] `regional einkaufen` (Germany)

#### Team Member 5
- [ ] `bio lieferung` (Germany)
- [ ] `grillen rezepte` (Germany)
- [ ] `salat rezepte` (Germany)

#### Team Member 6
- [ ] `bio company berlin` (Berlin)
- [ ] `smoothie rezepte` (Germany)
- [ ] `zero waste` (Germany)

**How to download:**
1. Go to https://trends.google.com/trends/
2. Search keyword → Set location (Germany/Berlin) → Time: 5 years
3. Click ↓ download button (top-right of chart)
4. Save as: `trends_KEYWORD_de_5y.csv`
5. Upload to GitHub: `data/raw/google_trends/`

---

## 📈 Next Phase: Forecasting Model

### Once we have Google Trends data:

```
Week 1: Data Integration
├── Merge all data sources (script ready)
├── Create lag features (1-4 weeks)
├── Validate data quality
└── Output: agricom_unified_weekly.csv

Week 2: Baseline Model
├── Train Prophet/ARIMA model
├── Test demand-signal correlations
├── Generate preliminary forecasts
└── Output: baseline_forecast.csv

Week 3: Refinement
├── Neighborhood-specific models
├── Seasonal adjustments
├── Confidence intervals
└── Output: segment_forecasts.csv
```

---

## 🔧 New Scripts Available

### 1. `google_trends_serpapi.py`
- Alternative collection method (bypasses rate limits)
- Requires SerpAPI key (optional)
- Has `--status` flag to check collection progress
- Has `--guide` flag to generate download instructions

### 2. `economic_indicators.py`
- Fetches OECD Consumer Confidence Index
- Fetches Eurostat food price data
- Fetches World Bank economic indicators
- No API key required

### 3. `social_signals.py`
- YouTube trending analysis
- Reddit organic food discussions
- Instagram hashtag tracking (needs RapidAPI key)

### 4. `data_merger.py`
- Combines ALL data sources into unified dataset
- Creates both daily and weekly versions
- Adds lag and rolling features automatically

---

## 📋 Meeting Agenda Suggestion

### 1. Status Update (10 min)
- Review collected data
- Confirm Google Trends assignments

### 2. Data Quality Discussion (10 min)
- Any issues with collected data?
- Need different time ranges?

### 3. Hypotheses Refinement (15 min)
- Which signals show most promise?
- Any new hypotheses from team?

### 4. Timeline Review (10 min)
- Midpoint presentation date
- Who presents what?

### 5. Action Items (5 min)
- Confirm Google Trends deadline
- Reddit API credentials?
- Next meeting time

---

## 📞 Questions for AgriCom/Mentor

1. **Sales Data:** Do we need actual AgriCom sales data for validation, or is methodology demonstration sufficient?

2. **Geographic Scope:** Should we focus on one Berlin neighborhood or all three?

3. **Forecast Granularity:** What granularity does AgriCom need? (Daily/Weekly/Monthly)

4. **Product Categories:** Should we narrow to specific products (tomatoes, salads) or keep it broad (all organic)?

---

## 🗂️ Repository Structure

```
ACP-AGRICOM/
├── data/
│   ├── raw/
│   │   ├── weather_berlin_20260122.csv ✅
│   │   ├── events_berlin_20260122.csv ✅
│   │   ├── gdelt_articles_20260122.csv ✅
│   │   ├── gdelt_timeline_20260122.csv ✅
│   │   └── google_trends/
│   │       ├── bio_gemuese_de.csv ✅
│   │       ├── bio_tomaten_de.csv ✅
│   │       ├── ... (need more) ⚠️
│   └── processed/ (empty - run data_merger.py)
├── src/
│   ├── data_collection/
│   │   ├── weather.py ✅
│   │   ├── events.py ✅
│   │   ├── gdelt_news.py ✅
│   │   ├── google_trends_serpapi.py 🆕
│   │   ├── economic_indicators.py 🆕
│   │   └── social_signals.py 🆕
│   └── analysis/
│       ├── weather_analysis.py ✅
│       ├── gdelt_analysis.py ✅
│       ├── neighborhood_segmentation.py ✅
│       ├── trends_weather_analysis_v2.py ✅
│       └── data_merger.py 🆕
└── outputs/
    ├── figures/ (13 visualizations) ✅
    └── reports/ (5 reports) ✅
```

---

## ✅ Ready to Run Commands

```bash
# Navigate to project
cd /path/to/ACP-AGRICOM

# Check Google Trends status
python src/data_collection/google_trends_serpapi.py --status

# Generate download guide for team
python src/data_collection/google_trends_serpapi.py --guide

# Collect economic indicators
python src/data_collection/economic_indicators.py

# After team uploads trends, merge all data
python src/analysis/data_merger.py
```

---

*Document prepared for AGRICOM team meeting*
*Contact: Matias Cam*
