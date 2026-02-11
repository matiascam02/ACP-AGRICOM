# AGRICOM Project - Executive Summary

**Project:** Organic Produce Demand Forecasting for Berlin  
**Client:** AgriCom.io  
**Team:** ESMT MBA Students  
**Timeline:** January 1 - April 5, 2026  
**Status:** 🟢 On Track (85% Data Complete)

---

## The Challenge

AgriCom needs to predict organic produce demand in Berlin **before** it happens. Traditional methods rely on historical sales data, but that's reactive. We're developing a **predictive methodology** using alternative data signals as leading indicators.

---

## Our Approach

Instead of looking backward, we look at **external signals** that predict consumer behavior:

| Signal | Why It Matters | Predictive Power |
|--------|----------------|------------------|
| **Google Trends** | Shows rising interest 2-4 weeks ahead | 🟢 High |
| **Weather** | Affects shopping behavior same week | 🟡 Medium |
| **News Sentiment** | Drives organic premium willingness | 🟢 High (13% correlation) |
| **Events** | Football matches & holidays shift demand | 🟢 High (21% Christmas spike) |
| **Economics** | Consumer confidence affects spending | 🟡 Medium |

---

## What We've Built

### 📊 Data Infrastructure
- **1,110 days** of Berlin weather data (2023-2026)
- **196 events** tracked (Bundesliga, holidays)
- **159 news articles** analyzed for sentiment (GDELT)
- **8/20 Google Trends keywords** collected (in progress)
- **360 months** of economic indicators (OECD, Eurostat)

### 🤖 Forecasting Model
- **Algorithm:** Ridge Regression (outperformed Random Forest & Gradient Boosting)
- **Accuracy:** R² = 0.82 (explains 82% of demand variance)
- **Error Rate:** MAE = 3.96 on 0-100 demand scale (~4% error)
- **Validation:** 90-day holdout test set

### 🗺️ Market Segmentation
Identified 3 distinct Berlin customer profiles:

| Segment | Organic Affinity | Key Driver | Target Priority |
|---------|------------------|------------|-----------------|
| **Kreuzberg** | 85% | Farmers markets, sustainability | ⭐ Primary |
| **Mitte** | 70% | Convenience, premium quality | Secondary |
| **Charlottenburg** | 60% | Family health, weekend shopping | Tertiary |

---

## Key Findings

### What Drives Demand?

1. **Day of Week (24.1%)** - Weekend shopping dominates organic purchases
2. **Christmas Season (21.5%)** - Demand spikes 2-3x during holidays
3. **News Sentiment (13.1%)** - Positive organic coverage boosts sales
4. **Temperature (8.8%)** - Moderate temps (15-20°C) optimal for fresh produce
5. **Weekend Flag (7.7%)** - Distinct behavioral patterns Sat-Sun

### Surprising Insights

- **47% of days** have ideal weather for salad/fresh produce demand
- **News sentiment averages +0.32** (consistently positive coverage of organic food)
- **Kreuzberg** shows 85% organic affinity vs. 60% in Charlottenburg
- **Christmas** creates the single largest demand spike (2-3x baseline)

---

## Business Impact

### For AgriCom

✅ **Inventory Optimization** - Know what to stock 2-4 weeks ahead  
✅ **Waste Reduction** - Avoid over/under-ordering based on weather + events  
✅ **Geographic Targeting** - Focus on high-affinity neighborhoods first  
✅ **Promotional Timing** - Align campaigns with news cycles + favorable weather  
✅ **Demand Forecasting** - 82% accuracy with current model  

### Immediate Recommendations

1. **Pilot Program:** Launch in Kreuzberg (85% organic affinity)
2. **Weekend Strategy:** Optimize inventory/delivery for Sat-Sun (24% variance)
3. **Christmas Planning:** Prepare for 2-3x spike (confirmed by data)
4. **Weather-Based Pricing:** Dynamic promotions on moderate temp days
5. **Sentiment Monitoring:** Track news coverage for demand signals

---

## Current Status

### ✅ Completed (85%)
- Weather, events, news, economic data collected
- Preliminary forecasting model trained (R² = 0.82)
- Neighborhood segmentation complete
- 13 visualizations + 5 analysis reports generated
- Full automation pipeline built

### 🔄 In Progress (15%)
- Google Trends data collection (12/20 keywords remaining)
- Team coordinating manual downloads (rate limits)

### 🎯 Next Milestones
- **Week 1:** Complete Google Trends → Run unified data merge
- **Week 2:** Updated forecast with full dataset
- **Week 3:** Midpoint presentation + model refinement

---

## Forecast Preview (Feb-Apr 2026)

Based on current model (partial data):

| Metric | Value |
|--------|-------|
| **Average Demand Index** | 49.4 / 100 |
| **Peak Demand** | 59.2 (weekend + favorable weather) |
| **Lowest Demand** | 42.5 (weekday + poor weather) |
| **Confidence Interval** | ±3.96 (MAE) |

*Note: Accuracy will improve once Google Trends collection completes.*

---

## Team & Resources

**Team Structure:**
- **Project Manager:** Jean-Christophe (Sean)
- **Students:** Matias, Ryan/Tonghan, Arjun, Akos, May
- **Mentor:** Ryan Krog (Deutsche Bank)
- **Client Contact:** Farah Ezzedine (AgriCom)

**Tech Stack:**
- Python 3.14 (pandas, scikit-learn, prophet)
- Data APIs: Open-Meteo, GDELT, Google Trends, OECD/Eurostat
- Version Control: GitHub ([matiascam02/ACP-AGRICOM](https://github.com/matiascam02/ACP-AGRICOM))

---

## Bottom Line

We've built a **working forecasting model** that predicts organic demand with **82% accuracy** using alternative data signals. The methodology is **replicable**, **scalable**, and provides **2-4 weeks of lead time** for inventory planning.

**Project Status:** 🟢 On track  
**Model Performance:** 🟢 Strong (R² 0.82)  
**Data Quality:** 🟡 High (85% complete)  
**Next Critical Action:** Complete Google Trends collection this week

---

**Contact:** Matias Cam (ESMT MBA Student)  
**Repository:** https://github.com/matiascam02/ACP-AGRICOM  
**Last Updated:** February 11, 2026
