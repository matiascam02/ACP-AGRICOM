# Methodological Blueprint
**Project:** Organic Produce Demand Forecasting — AgriCom.io  
**Prepared by:** ESMT ACP Team  
**Date:** February 18, 2026  
**Version:** 1.0 — For Client Review

---

## Executive Summary

This document formalizes the methodological approach for forecasting organic produce demand in Berlin using alternative data signals. Our goal is to deliver a transparent, academically-grounded, and operationally viable forecasting framework that AgriCom can validate before we proceed to full modelling.

**Two decisions we need from you:**
1. Should the framework be scoped to Berlin only, or designed to generalize to additional cities?
2. Can you share the data schema (column names/definitions) or a small anonymized sample to validate our unit-of-analysis assumptions?

---

## 1. Problem Framing & Objectives

### What We Are Solving Now
Predict **organic produce demand at the city/neighborhood level in Berlin**, 1–4 weeks ahead, using alternative external signals (weather, events, news sentiment, search trends, economic indicators) as leading indicators.

**Why alternative signals?** AgriCom currently lacks access to historical point-of-sale data. External signals serve as demand proxies that are observable before sales occur — allowing a predictive (not reactive) approach.

### What We Are Not Solving Yet
- SKU-level forecasting (requires product-level sales data)
- Real-time demand tracking (requires live POS integration)
- Price elasticity modelling (requires pricing history)
- Supply-side constraints (requires supplier data)

These are natural extensions once a demand signal baseline is established.

---

## 2. Demand Definition (Dependent Variable)

### Unit of Analysis
| Dimension | Definition |
|-----------|------------|
| **Product** | Organic produce category (aggregate, not SKU-level) |
| **Time** | Daily granularity (aggregatable to weekly) |
| **Location** | Berlin neighborhood (Kreuzberg, Mitte, Charlottenburg as primary segments) |

### Demand Proxy Construction
Since no direct sales data is available, demand is operationalized as a **composite demand index (0–100 scale)** derived from:
- Google Trends search volume for 20 organic food keywords (e.g., "Bio Gemüse Berlin", "organic food delivery")
- Normalized to a 0–100 scale using min-max normalization across the observation window
- Aggregated weekly to smooth noise

**Aggregation logic:**
```
Demand Index (daily) = weighted average of normalized Google Trends keywords
Weights: search volume share per keyword (updated quarterly)
```

**Limitation and validation path:** Once AgriCom shares sales data (even anonymized), we will re-calibrate the index as the true dependent variable. The current proxy has high face validity (search precedes purchase) and is consistent with academic literature on nowcasting consumer behavior.

---

## 3. Variable Architecture (Independent Variables / Signals)

### 3.1 Signal List with Relevance Rationale

| Signal | Source | Hypothesized Direction | Lag Assumption | Feature Importance (current model) |
|--------|--------|----------------------|----------------|--------------------------------------|
| **Day of week** | Engineered | Weekend ↑ demand | Same day | 24.1% |
| **Christmas/holiday season** | Events calendar | Seasonal ↑ 2–3x | 0–7 days | 21.5% |
| **News sentiment (organic food)** | GDELT | Positive tone ↑ demand | 3–7 days | 13.1% |
| **Temperature (mean)** | Open-Meteo | Moderate temp (15–20°C) ↑ fresh produce | Same day | 8.8% |
| **Weekend flag** | Engineered | ↑ demand Sat–Sun | Same day | 7.7% |
| **Precipitation** | Open-Meteo | Rain ↓ foot traffic → ↓ demand | Same day | ~4% |
| **Google Trends (organic keywords)** | Google Trends | Rising searches ↑ demand 2–4 weeks ahead | 14–28 days | ~6% |
| **Bundesliga home matches** | OpenLigaDB | ↓ organic demand (competing activity) | Same day | ~3% |
| **Consumer confidence** | OECD/Eurostat | High confidence ↑ premium spending | 30–60 days | ~2% |
| **Food price inflation** | Eurostat CPI | High inflation ↓ premium organic spend | 30–60 days | ~2% |

### 3.2 Hypothesized Interaction Effects

| Interaction | Expected Effect |
|-------------|----------------|
| Weekend × Good weather | Amplified demand spike (farmers markets, outdoor shopping) |
| Christmas season × Google Trends spike | Compound demand peak |
| Rain × Weekday | Strongest demand suppression |
| High news sentiment × Weekend | Premium purchase behavior reinforcement |

These interactions will be tested via inclusion of product terms in the regression and via partial dependence plots in the Random Forest model.

### 3.3 Variables Excluded and Why

| Excluded Variable | Reason |
|------------------|--------|
| Social media (Instagram, Reddit, YouTube) | High collection complexity, low marginal R² gain vs. cost |
| Competitor promotions | No accessible data source |
| Store-level foot traffic | Requires proprietary data (potential future integration) |

---

## 4. Data Requirements & Sources

### 4.1 Internal Data Fields Needed from AgriCom
*(Structure only — we do not require the actual data at this stage)*

| Field | Type | Description |
|-------|------|-------------|
| `date` | date | Transaction date (daily or weekly) |
| `location` | string | Store/neighborhood identifier |
| `product_category` | string | Organic produce category |
| `quantity_sold` | numeric | Units or kg sold |
| `revenue` | numeric | Optional — for value-weighted demand |

**Minimum viable schema:** `date`, `location`, `quantity_sold` at daily granularity.

### 4.2 External Data Sources

| Source | Data | Update Frequency | Access |
|--------|------|-----------------|--------|
| Open-Meteo | Weather (Berlin, 2023–2026) | Daily | Free API |
| GDELT | News sentiment (Germany) | Sub-daily | Free API |
| Google Trends | Organic keyword searches | Weekly | Manual / SerpAPI |
| OpenLigaDB | Bundesliga match schedule | Per season | Free API |
| OECD / Eurostat | Consumer confidence, CPI | Monthly | Free API |

**Current data status:**
- Weather: ✅ 1,110 days (2023–2026)
- Events: ✅ 196 events
- GDELT Sentiment: ✅ 159 articles, 2,856 timeline points
- Google Trends: ⚠️ 8/20 keywords collected (rate-limited, manual completion in progress)
- Economic Indicators: ✅ 360 months

---

## 5. Validation Plan

### 5.1 Evaluation Approach
We explicitly reject in-sample accuracy as a success metric. All model evaluation uses **out-of-sample holdout testing**.

| Approach | Description |
|----------|-------------|
| **Holdout split** | Last 90 days reserved for testing (not used in training) |
| **Time-series cross-validation** | Rolling window CV to simulate real forecasting conditions |
| **Walk-forward validation** | Model retrained monthly on expanding window |

### 5.2 Performance Metrics
| Metric | Current (Ridge, proxy data) | Target (with real data) |
|--------|-----------------------------|-----------------------|
| R² | 0.82 | > 0.75 |
| MAE | 3.96 (0–100 scale) | TBD post data integration |
| RMSE | TBD | TBD |
| Directional accuracy | TBD | > 70% |

### 5.3 Baseline Comparison
All models are compared against:
1. **Naïve baseline:** Last week's demand = this week's demand
2. **Seasonal baseline:** Same week last year
3. **Moving average (4-week)**

A model that does not outperform these baselines on holdout data will not be advanced to production.

---

## 6. Data Quality Handling

### 6.1 Missing Values
| Variable | Strategy |
|----------|----------|
| Weather (rare gaps) | Linear interpolation for ≤3 consecutive days; flag for >3 |
| Google Trends (rate limits) | Forward fill weekly values to daily; flag imputed rows |
| GDELT (no data periods) | Treat as neutral sentiment (tone = 0); flag rows |
| Events (no event = 0) | Binary/categorical encoding; absence is meaningful |

### 6.2 Outliers
| Method | Application |
|--------|-------------|
| IQR-based flagging | Applied to continuous variables (temperature, precipitation) |
| Manual inspection | Christmas, COVID-period anomalies reviewed manually |
| Winsorizing | Applied at 1st/99th percentile for economic indicators |

Outliers are flagged but **not removed** by default — they are handled via robust regression techniques (Ridge, Huber loss) that are less sensitive to extreme values than OLS.

### 6.3 Consistency Checks
- Date continuity: assert no gaps in time series
- Value ranges: assert weather variables within physical bounds
- Schema validation: automated checks at data ingestion

---

## 7. Literature Anchor

The following academic references underpin our methodological choices:

| Reference | Relevance |
|-----------|-----------|
| Choi & Varian (2012). "Predicting the Present with Google Trends." *Economic Record.* | Theoretical basis for search volume as demand nowcast |
| Goel et al. (2010). "Predicting consumer behavior with Web search." *PNAS.* | Validation of search-based demand forecasting |
| Makridakis et al. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." *International Journal of Forecasting.* | Benchmark methodology for time series; Ridge competitive at this scale |
| Hoerl & Kennard (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics.* | Foundation for Ridge Regression choice (handles multicollinearity in our signal set) |
| Tetlock & Gardner (2015). *Superforecasting.* | Calibration principles applied to model evaluation and uncertainty quantification |

---

## 8. Open Questions for AgriCom

To make this framework maximally useful, we need your input on two questions:

**Q1: Geographic Scope**
> Should the initial framework be scoped to **Berlin only**, or designed to **generalize to additional cities** from the start?

Our recommendation: start Berlin-first to ensure signal relevance, then extend. But if you plan to expand to Hamburg or Munich within 12 months, we should design the pipeline for multi-city from the start (adds ~20% complexity).

**Q2: Data Schema**
> Can you confirm the data schema (column names/definitions) or share a small anonymized sample (e.g., 5–10 rows, no actual values needed)?

This allows us to validate our unit-of-analysis assumptions before building the full pipeline. Misaligned granularity (e.g., weekly vs. daily) would require structural changes to the model.

---

*Document prepared by ESMT ACP Team for AgriCom.io review.*  
*Next milestone: Methodological alignment meeting — EOD Monday, February 23, 2026.*
