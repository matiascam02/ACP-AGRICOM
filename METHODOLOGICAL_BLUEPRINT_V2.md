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

**Note:** This version proposes a specific product basket with rationale. We look forward to your feedback on our approach.

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

Since no direct sales data is available, demand is operationalized as a **weekly composite demand index** for the "Salad Essentials" basket (0-100 scale):

```
Basket Demand Index (weekly) = weighted average of:
- Google Trends: "bio tomaten", "bio salat", "bio gurken", "bio paprika Berlin"
- Weighted by product share (35% tomatoes, 30% leafy greens, 20% cucumbers, 15% peppers)
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

### 2.2 Proposed Product Basket

Based on Nadim's recommendation to *"create a profile of three, four crops that are used regularly"*, we propose the following basket:

#### **The "Salad Essentials" Basket**

We propose focusing on **4 products that form a coherent "salad bundle"**. Here's the reasoning for each selection:

---

##### **Product 1: Tomatoes (Standard/Round)**

**Why included:** Primary anchor product

| Criterion | Tomatoes | Decision Factor |
|-----------|----------|-----------------|
| **Organic WTP** | Highest in category | Consumers pay premium for taste difference |
| **Purchase frequency** | Weekly | Regular salad ingredient drives consistent demand |
| **Use cases** | Multiple | Salads, cooking, sandwiches = broad demand base |
| **Perishability** | 5-7 days | High turnover = more data points |
| **Price volatility** | Moderate | Good signal-to-noise ratio |

**Why standard tomatoes (not cherry):** Cherry tomatoes serve a niche (garnishing/snacking) with lower volume and different seasonality. Standard tomatoes represent the mass market and have more stable demand patterns.

---

##### **Product 2: Leafy Greens (Mixed Salad/Bagged)**

**Why included:** Primary complement to tomatoes

| Criterion | Leafy Greens | Decision Factor |
|-----------|--------------|-----------------|
| **Co-purchase rate** | High with tomatoes | 60%+ of tomato buyers also buy salad mix |
| **Perishability** | 3-5 days | Even higher turnover than tomatoes |
| **Price point** | Premium | Similar organic WTP profile to tomatoes |
| **Seasonality** | Counter-cyclical | Greenhouse production different from field crops |
| **Consumer overlap** | Strong | Same health-conscious demographic |

**Why bagged mix (not whole heads):** Pre-washed bagged salads dominate organic purchases and represent "convenience organic" — the growing segment AGRICOM serves.

---

##### **Product 3: Cucumbers**

**Why included:** Secondary complement, adds signal diversity

| Criterion | Cucumbers | Decision Factor |
|-----------|-----------|-----------------|
| **Use case** | Salad essential | Classic combination with tomatoes |
| **Growing season** | Summer peak | Different from tomatoes (year-round) |
| **Price elasticity** | High | Sensitive to promotions and weather |
| **Substitutability** | Medium | Can substitute for peppers in salads |
| **Volume** | Consistent | Steady demand throughout year |

**Signal value:** Cucumbers provide **weather sensitivity contrast** — they're more summer-dependent than tomatoes (which are greenhouse-grown year-round). This helps the model separate true weather effects from noise.

---

##### **Product 4: Bell Peppers (Red/Yellow)**

**Why included:** Tertiary component, price-tier diversity

| Criterion | Bell Peppers | Decision Factor |
|-----------|--------------|-----------------|
| **Price point** | Premium | Higher than tomatoes/cucumbers |
| **Use case** | Salad + cooking | Versatile = stable demand |
| **Color signal** | Red/yellow | "Premium organic" visual cue |
| **Seasonality** | Distinct | Different harvest windows |
| **Cross-elasticity** | High | Substitutes with cucumbers in recipes |

**Why peppers (not carrots/onions):** Peppers share the "raw salad" use case with tomatoes/greens/cucumbers. Root vegetables (carrots) and aromatics (onions) have different purchase patterns and lower organic premiums.

---

#### **Why This Specific Combination?**

**1. The "Salad Bundle" Behavioral Cluster**

These 4 products are purchased together as a meal solution:
- Consumer need: "I want to make a salad"
- Basket items: Tomatoes + Greens + Cucumber + Peppers
- Result: Co-movement in demand provides modeling stability

**2. Cross-Elasticity Triangle**

```
        Tomatoes
       /        \
  Cucumbers — Bell Peppers
       \
       Leafy Greens
```

- **Substitution:** If tomatoes expensive → more peppers/cucumbers
- **Complementarity:** All 4 rise together on "salad days" (weekends, good weather)
- **Price tier coverage:** Low (cucumbers) → Medium (tomatoes) → High (peppers)

**3. Signal Diversity for Robustness**

| Product | Primary Driver | Secondary Driver |
|---------|---------------|------------------|
| Tomatoes | Health trends | Cooking use |
| Greens | Convenience | Perceived freshness |
| Cucumbers | Summer weather | Price promotions |
| Peppers | Recipe trends | Color/visual appeal |

Different drivers = model doesn't rely on single signal type

**4. Noise Normalization**

Individual shocks are smoothed:
- Morocco tomato shortage → peppers/cucumbers compensate
- E. coli lettuce recall → other products still in basket
- Cucumber price spike → tomatoes carry weight

**5. AGRICOM Actionability**

This basket represents **core organic produce** that:
- Moves in consistent volumes
- Has established supply chains
- Represents inventory planning priority
- Can be promoted as "salad bundle"

---

#### **Basket Construction**

**Basket Demand Index =**
- **35% Tomatoes** (highest volume, anchor product)
- **30% Leafy Greens** (highest frequency, co-purchase driver)
- **20% Cucumbers** (signal diversity, seasonal contrast)
- **15% Bell Peppers** (premium tier, price elasticity data)

*Weights based on typical organic produce basket share and revenue contribution*

### 2.3 Products Considered but Excluded

We evaluated **12 candidate products** before selecting the final 4. Here's the decision logic:

#### **Excluded: Cherry Tomatoes**

| Factor | Assessment | Decision |
|--------|-----------|----------|
| Market size | 15% of tomato category | Too small |
| Use case | Garnish/snacking vs. cooking | Different behavior |
| Data quality | Higher variance, more noise | Unreliable signal |
| **Verdict** | | **EXCLUDED** — niche product, better as Phase 2 extension |

#### **Excluded: Root Vegetables (Potatoes, Carrots, Onions)**

| Factor | Assessment | Decision |
|--------|-----------|----------|
| Use case | Cooking staples vs. raw salad | Different purchase occasion |
| Organic WTP | Lower — taste difference less obvious | Weak organic signal |
| Shelf life | 2-4 weeks | Too long, reduces purchase frequency |
| **Verdict** | | **EXCLUDED** — belong to different demand cluster |

#### **Excluded: Fruits (Berries, Apples, Bananas)**

| Factor | Assessment | Decision |
|--------|-----------|----------|
| Consumption | Snacking/dessert vs. meal prep | Different behavioral trigger |
| Seasonality | Extreme peaks (berries) | Too volatile for stable model |
| **Verdict** | | **EXCLUDED** — require separate fruit category model |

#### **Excluded: Cooking Vegetables (Zucchini, Eggplant, Broccoli)**

| Factor | Assessment | Decision |
|--------|-----------|----------|
| Use case | Cooked meals only | Don't fit "raw salad" cluster |
| Purchase pattern | Recipe-driven, lumpy | Less predictable than salad items |
| **Verdict** | | **EXCLUDED** — different demand drivers |

#### **Excluded: Herbs (Basil, Parsley, Cilantro)**

| Factor | Assessment | Decision |
|--------|-----------|----------|
| Volume | <2% of produce sales | Too small for forecasting |
| Price volatility | Extreme (supply shocks) | Unreliable signal |
| **Verdict** | | **EXCLUDED** — insufficient data volume |

---

#### **Decision Summary: Why 4 Products?**

| Basket Size | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **1 product** (tomatoes only) | Cleanest signal | No cross-elasticity; high noise sensitivity | Too narrow |
| **2 products** | Simple | Limited signal diversity; fragile to shocks | Insufficient |
| **3 products** | Balanced | May still miss key substitution patterns | Borderline |
| **4 products** | Optimal diversity; captures salad bundle | Manageable complexity | **SELECTED** |
| **5+ products** | Maximum robustness | Diminishing returns; complexity increases | Too broad |

**4 products captures the "salad bundle" comprehensively** without adding noise from unrelated categories.

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
| 5 | Google Trends (basket keywords) | Google Trends | Rising searches ↑ demand | 1-3 weeks | ~6% |
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
| **price_organic** | numeric | EUR/kg - organic (per basket product) | **HIGH** |
| **price_conventional** | numeric | EUR/kg - conventional (per basket product) | **HIGH** |
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

For weekly basket demand:

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

**Q1: Pricing Data**
> Can you confirm what pricing data is available for the basket products?
> - Own-prices (organic: tomatoes, leafy greens, cucumbers, peppers)
> - Substitute prices (conventional equivalents)
> - Promotion history / discount patterns

**Q2: Data Granularity**
> What is the finest granularity you can share? Weekly by store/neighborhood, or only aggregated?

**Q3: Consumer Data**
> Is any customer segmentation data available, or should we use demographic proxies by neighborhood?

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
