# Methodological Blueprint v2.0

**Project:** Organic Produce Demand Forecasting — AgriCom.io  
**Prepared by:** ESMT ACP Team  
**Date:** February 24, 2026  
**Version:** 2.0 — Revised based on client feedback

---

## Executive Summary

This document presents a revised methodological approach based on feedback from Nadim (AGRICOM) and Vlada (ESMT). The key changes from v1.0:

1. **Narrowed product scope** — Focus on tomatoes (and complementary basket)
2. **Weekly granularity** — Switched from daily to weekly demand
3. **Consumer segmentation** — Added structured consumer profiles
4. **Price framework** — Explicitly incorporated as primary demand driver

**Two decisions we still need from AGRICOM:**

1. Confirm product scope (tomatoes primary, or basket approach)
2. Provide pricing data schema (or confirm what data is available)

---

## 1. Demand Definition (Revised)

### 1.1 Unit of Analysis

| Dimension | Definition (v2.0) |
|-----------|-------------------|
| **Product** | Tomatoes (primary) + optional basket of 3-4 complementary crops |
| **Time** | **Weekly** granularity (changed from daily) |
| **Location** | Berlin neighborhood level |

### 1.2 Why Weekly Instead of Daily

As Nadim pointed out: *"If you want to look at daily, you have to integrate the shelf life. So they bought it when? When was the last time that they bought? Then there's still left or not?"*

Weekly aggregation:
- Avoids shelf-life complexity
- Reduces noise from daily fluctuations
- Aligns with typical purchasing patterns for produce
- Simpler to model and validate

### 1.3 Demand Proxy Construction

Since no direct sales data is available, demand is operationalized as a **weekly composite demand index** for tomatoes (0-100 scale):

```
Demand Index (weekly) = weighted average of:
- Google Trends: "bio tomaten", "organic tomatoes Berlin"
- Normalized to 0-100 scale
- Aggregated weekly (Mon-Sun)
```

---

## 2. Product Scope

### 2.1 Why Tomatoes?

Based on Nadim's feedback: *"You cannot treat all crops the same because certain crops, let's say a tomato that you use in a salad, you have a higher willingness of a certain customer to pay to buy organic because of like the taste and so on."*

| Characteristic | Tomatoes | Rationale |
|----------------|----------|-----------|
| **Perishability** | High | Short shelf life → frequent repurchase |
| **Purchase frequency** | Weekly | Regular salad ingredient |
| **Price elasticity** | High | Sensitive to organic premium |
| **Consumer segment** | Broad | Both premium buyers and price-sensitive |
| **Seasonality** | Moderate | Year-round with seasonal peaks |
| **Use case** | Multiple | Fresh consumption, cooking, salads |

### 2.2 The "Basket" Option

As Nadim suggested: *"I would create a profile of three, four crops that are used regularly. And then you would say who are the actual people that will buy this type of crops regularly."*

**Recommended Basket (v2.0):**

1. **Tomatoes** — Primary focus
2. **Leafy greens** (salad mix) — Complementary, similar purchase pattern
3. **Cucumbers** — Often consumed together in salads

This basket:
- Represents a typical "salad bundle"
- Normalizes noise from individual crop price fluctuations
- Captures cross-product substitution effects

### 2.3 Alternative: Cherry Tomatoes vs. Regular Tomatoes

As Nadim noted: *"I would try to look at the different type of tomatoes, being it processed, being it cherry tomatoes or the normal one and see if there is something compared between them."*

**Scope for v2.0:** Focus on standard tomatoes first; cherry tomatoes as future extension.

---

## 3. Consumer Segmentation

### 3.1 Why Segmentation Matters

As Nadim emphasized: *"Certain customer types pursue bio tomatoes. And this will be completely different types of customers who pursue non-bio tomatoes who say I don't want to pay more for bio tomatoes."*

Different consumer segments have:
- Different price sensitivities
- Different weather sensitivities (urban vs suburban)
- Different event sensitivities (families vs young professionals)
- Different search behaviors

### 3.2 Proposed Consumer Segments

| Segment | Profile | Characteristics | Price Sensitivity | Key Drivers |
|---------|---------|-----------------|-------------------|-------------|
| **Premium Sustainability Buyers** | Higher income, 25-45, urban (Kreuzberg, Mitte) | Will pay premium for organic, value sustainability | Low | Health news, influencers, brand |
| **Price-Sensitive Families** | Families with kids, suburban (Lichtenberg, Pankow) | Seek value, bulk purchases | High | Price promotions, discounts |
| **Convenience-Driven Urban** | Young professionals, single/households | Quick shop, quality over price | Medium | Proximity, delivery options |
| **Traditional Cooks** | 45+, all neighborhoods | Cook at home, brand loyalty | Medium | Seasonality, recipe trends |

### 3.3 Segmentation Strategy

**Recommendation:** Segmentation should **precede** modeling, not follow it.

As Nadim asked: *"Different neighborhoods may indeed prefer different types."*

Approach:
1. Use neighborhood demographics to proxy segment composition
2. Build segment-specific signal weights
3. Aggregate for overall market forecast

---

## 4. Price Framework (NEW SECTION)

### 4.1 Why Price Must Be Included

As Vlada emphasized: *"Price is typically the primary predictor of demand. In almost any product category, price (and relative price) plays a central role and must be explicitly incorporated."*

**Without price modeling:**
- Inflation-driven demand shifts may be incorrectly attributed to weather/search
- Promotion-driven spikes may be interpreted as seasonal effects
- Substitution effects may distort external signal relevance

### 4.2 Price Components

| Component | Definition | Data Source |
|-----------|------------|-------------|
| **Own-price** | Organic tomato price (EUR/kg) | AgriCom internal |
| **Substitute price** | Conventional tomato price | Market data |
| **Complementary price** | Prices of related basket items (lettuce, cucumber) | Market data |
| **Relative price gap** | Organic premium vs conventional (%) | Calculated |
| **Promotions** | Temporary price reductions | AgriCom internal |

### 4.3 Price Exogeneity Treatment

As Vlada requested, we clarify the treatment:

**Recommended: Controlled (minimum acceptable)**

- Include price as a control variable to avoid omitted variable bias
- Not the main object of study, but necessary for signal isolation

**Why not endogenous?**
- Organic pricing often set by contract/central negotiation
- Weekly granularity reduces feedback loop complexity

**Implication:** We can include price as a predictor without worrying about reverse causality in this framework.

### 4.4 How Price Separates Signal from Noise

Example from Vlada:
> *"If organic tomato price rises relative to conventional tomatoes, and demand falls: Is that weather? Is that sentiment? Or simply substitution?"*

With explicit price modeling:
- Separate own-price elasticity from weather effects
- Identify substitution (organic → conventional)
- Isolate signal-driven demand from purchasing power effects
- Control for promotion-driven spikes

---

## 5. Updated Signal Architecture

### 5.1 Revised Signal Priority

Based on client feedback, the signal importance hierarchy is revised:

| Rank | Signal | Source | Rationale |
|------|--------|--------|-----------|
| 1 | **Price (own + relative)** | AgriCom + Ma}
### 5.2 Complete Signal List (Revised)

| Rank | Signal | Source | Hypothesized Direction | Lag | Feature Importance |
|------|--------|--------|----------------------|-----|-------------------|
| 1 | Price (own + relative) | AgriCom + Market | Inverse to demand | Same week | ~30% (expected) |
| 2 | Day of week | Engineered | Weekend ↑ demand | Same day | ~20% |
| 3 | Christmas/holiday season | Events calendar | Seasonal ↑ 2-3x | 0-7 days | ~15% |
| 4 | Temperature (mean) | Open-Meteo | Moderate temp ↑ fresh | Same day | ~8% |
| 5 | Google Trends (tomato keywords) | Google Trends | Rising searches ↑ demand | 1-3 weeks | ~6% |
| 6 | News sentiment (organic food) | GDELT | Positive tone ↑ demand | 3-7 days | ~5% |
| 7 | Weekend flag | Engineered | ↑ demand Sat-Sun | Same day | ~5% |
| 8 | Precipitation | Open-Meteo | Rain ↓ foot traffic | Same day | ~4% |
| 9 | Bundesliga home matches | OpenLigaDB | ↓ demand (competing activity) | Same day | ~3% |
| 10 | Consumer confidence | OECD/Eurostat | High confidence ↑ premium spend | 2-4 weeks | ~2% |
| 11 | Food price inflation | Eurostat CPI | High inflation ↓ organic spend | 2-4 weeks | ~2% |

---

## 6. Data Requirements

### 6.1 Internal Data from AGRICOM (Revised)

| Field | Type | Description | Priority |
|-------|------|-------------|----------|
| date | date | Week (Monday-Sunday) | Required |
| product | string | "tomatoes" or "basket" | Required |
| location | string | Neighborhood/stores | Required |
| quantity_sold | numeric | Units or kg sold | Required |
| revenue | numeric | Optional | Medium |
| **price_organic** | numeric | EUR/kg - organic | **HIGH** |
| **price_conventional** | numeric | EUR/kg - conventional | **HIGH** |
| **promotion_flag** | binary | 1 if promoted | **HIGH** |
| customer_segment | string | Optional, if available | Low |

### 6.2 External Data Sources (Unchanged from v1.0)

| Source | Data | Status |
|--------|------|--------|
| Open-Meteo | Weather (Berlin, 2023-2026) | ✅ Ready |
| Events | Bundesliga + Holidays | ✅ Ready |
| GDELT | News sentiment | ✅ Ready |
| Google Trends | Tomato keywords | ⚠️ Partial (need tomato-specific) |
| OECD/Eurostat | Economic indicators | ✅ Ready |

---

## 7. Model Specification (Updated for v2.0)

### 7.1 Modeling Approach

For weekly tomato demand:

```
Demand_t = f(Price_t, Price_t-1, Temperature_t, Weekend_t, 
             Holiday_t, Trends_t, Sentiment_t, Events_t) + ε
```

### 7.2 Key Changes from v1.0

1. **Weekly aggregation** — Reduces noise, simplifies interpretation
2. **Price as primary predictor** — Controls for price effects before interpreting signals
3. **Tomato-specific Google Trends** — More relevant than generic "organic" terms
4. **Consumer segment weighting** — Adjust signals by segment composition

### 7.3 Expected Performance

| Metric | Target (with real data) |
|--------|------------------------|
| R² | > 0.75 |
| MAE | TBD (post data) |
| Directional accuracy | > 70% |

---

## 8. Open Questions for AGRICOM

**Q1: Product Scope**
> Should we focus on tomatoes only, or the basket approach (tomatoes + leafy greens + cucumbers)?

**Q2: Pricing Data**
> Can you confirm what pricing data is available?
> - Own-price (organic tomato)
> - Substitute price (conventional tomato)
> - Promotion history

**Q3: Consumer Data**
> Is any customer segmentation data available, or should we use demographic proxies?

---

## 9. Next Steps

1. **This week:** Send this revised blueprint to client
2. **Within 2 weeks:** Receive pricing data from AGRICOM
3. **Week 3:** Run model with price inclusion
4. **Week 4:** Consumer segmentation analysis

---

*Document prepared by ESMT ACP Team for AgriCom.io review.*  
*Version 2.0 — Incorporates feedback from Nadim (AGRICOM) and Vlada (ESMT)*  
*Previous version: METHODOLOGICAL_BLUEPRINT.md (v1.0, Feb 18, 2026)*
