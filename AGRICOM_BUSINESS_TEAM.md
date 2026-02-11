# AGRICOM Project - Business Team Section

**Team Role:** Presentation Storyline, Structure, Roles, Usecase, and Future Roadmap

---

## Presentation Storyline

### The Narrative Arc

**Act 1: The Problem (2-3 minutes)**
- AgriCom faces inventory challenges (waste, stockouts)
- Traditional methods are reactive (historical sales data)
- Need: Predictive approach with lead time

**Act 2: Our Solution (5-7 minutes)**
- Alternative data signals as leading indicators
- 2-4 weeks of advance warning
- Data + Model + Forecast pipeline

**Act 3: What We Built (8-10 minutes)**
- Data Team: 5 sources, comprehensive EDA
- Dev Team: 82% accuracy model, automated pipeline
- Business Team: Recommendations, usecase, roadmap

**Act 4: Impact & Next Steps (3-5 minutes)**
- Business value for AgriCom
- Pilot program proposal
- Future phases

**Total Time:** 20-25 minutes + 5-10 min Q&A

---

## Presentation Structure

### Slide Outline (30-35 slides)

#### Opening (3 slides)
1. **Title Slide**
   - AGRICOM Project: Organic Demand Forecasting
   - Team names, ESMT logo, AgriCom logo
   - Date

2. **Agenda**
   - Problem understanding
   - Our approach
   - Data & models
   - Results & recommendations
   - Next steps

3. **Team Introduction**
   - 6 students (names + roles)
   - Mentor: Ryan Krog
   - Client: Farah Ezzedine (AgriCom)

---

#### Problem & Context (3 slides)

4. **The Challenge**
   - Organic produce demand is volatile
   - High waste from over-ordering
   - Lost sales from under-ordering
   - Current methods: reactive, not predictive

5. **AgriCom's Need**
   - Optimize inventory 2-4 weeks ahead
   - Reduce waste, improve margins
   - Understand demand drivers (not just history)

6. **Our Hypothesis**
   - External signals predict demand before sales data
   - Google searches, weather, news, events, economics
   - Lead time = competitive advantage

---

#### Our Approach (4 slides)

7. **Methodology Overview**
   - Alternative data signals as leading indicators
   - Machine learning forecasting model
   - Neighborhood-specific segmentation

8. **Data Sources (Visual)**
   - 5 sources: Weather, Events, News, Trends, Economics
   - 3+ years coverage (2023-2026)
   - 85% complete (pending Google Trends)

9. **Project Workflow**
   - Data collection → Processing → Modeling → Forecasting
   - Fully automated pipeline
   - Weekly forecasts with confidence intervals

10. **Geographic Focus: Berlin**
    - 3 neighborhood segments
    - Kreuzberg (85% affinity) = pilot target
    - Scalable to other cities

---

#### Data Team (4 slides)

11. **Data Sources Explained**
    - Why we chose each source
    - Hypothesis behind each
    - Quality assessment

12. **EDA Highlights: Weather**
    - 47% warm days → salad/fresh produce
    - 34% rainy days → convenience shift
    - Seasonal patterns clear

13. **EDA Highlights: News Sentiment**
    - Average +0.32 (positive coverage)
    - "Bio-Lebensmittel sind so gefragt wie nie"
    - 13% correlation with demand

14. **EDA Highlights: Seasonality**
    - Christmas spike: 2-3x baseline
    - Weekend peaks: 24% variance
    - Google Trends confirm patterns

---

#### Dev Team (5 slides)

15. **Forecasting Model Overview**
    - 4 models tested: Ridge, RF, GBM, Prophet
    - Winner: Ridge Regression (R² = 0.82)
    - 90-day holdout validation

16. **Model Performance**
    - R² = 0.82 (explains 82% of variance)
    - MAE = 3.96 (~4% error on 0-100 scale)
    - Cross-validation: consistent across folds

17. **Feature Importance**
    - Top 5 drivers: Day of week, Christmas, Sentiment, Temp, Weekend
    - 98% explained by top 10 features
    - Interpretable for business stakeholders

18. **12-Week Forecast (Feb-Apr 2026)**
    - Average demand: 49.4
    - Peak: 59.2 (Easter weekend)
    - Low: 45.2 (early April weekday)
    - Chart: Forecast line + confidence interval

19. **Technical Architecture**
    - Python pipeline (pandas, scikit-learn)
    - Automated data collection
    - Reproducible, scalable code

---

#### Business Value (5 slides)

20. **Use Case: Inventory Optimization**
    - **Before:** Order based on last week's sales
    - **After:** Order based on 2-week forecast
    - **Result:** 15-20% waste reduction (estimate)

21. **Use Case: Promotional Planning**
    - **Before:** Generic weekly promotions
    - **After:** Target promotions on high-demand days
    - **Example:** Salad promotion on warm weekend

22. **Use Case: Geographic Targeting**
    - **Before:** Uniform distribution across Berlin
    - **After:** Prioritize Kreuzberg (85% affinity)
    - **Result:** Higher sales density in pilot area

23. **Use Case: Event-Based Stocking**
    - **Before:** Miss Christmas 2-3x spike
    - **After:** Prepare inventory 2 weeks ahead
    - **Result:** Capture holiday demand

24. **ROI Estimation (Hypothetical)**
    - Baseline: €100k monthly organic sales (AgriCom Berlin)
    - Waste reduction: 15% → €15k/month saved
    - Lost sales capture: 10% → €10k/month added
    - **Total:** €25k/month = €300k/year value

---

#### Recommendations (3 slides)

25. **Immediate Actions for AgriCom**
    1. Pilot program in Kreuzberg (Feb-Mar 2026)
    2. Integrate model into weekly planning
    3. Monitor news sentiment dashboard
    4. Optimize weekend inventory (24% variance)
    5. Prepare for Easter spike (April 20)

26. **Pilot Program Design**
    - **Location:** Kreuzberg farmers markets + stores
    - **Duration:** 8 weeks (Feb-Mar)
    - **Metrics:** Waste %, sales lift, forecast accuracy
    - **Success Criteria:** >10% waste reduction OR >5% sales lift

27. **Neighborhood Priority**
    - **Phase 1:** Kreuzberg (85% affinity)
    - **Phase 2:** Mitte (70% affinity)
    - **Phase 3:** Charlottenburg (60% affinity)
    - Roll out based on pilot results

---

#### Team Roles & Collaboration (2 slides)

28. **Team Structure**
    - **PM:** Jean-Christophe (Sean) - Coordination, timeline
    - **Data Team:** EDA, source selection, data quality
    - **Dev Team:** Modeling, forecasting, pipeline
    - **Business Team:** Storyline, recommendations, presentation
    - **Everyone:** Data collection (Google Trends)

29. **Collaboration Tools**
    - **GitHub:** Code + version control
    - **Obsidian:** Documentation + notes
    - **Slack/Email:** Communication
    - **Weekly Syncs:** Status updates, blockers

---

#### Future Roadmap (4 slides)

30. **Phase 2: Validation (Feb-Mar 2026)**
    - Complete Google Trends collection
    - Retrain model with full dataset (expect R² >0.85)
    - Test on actual AgriCom sales data (if provided)
    - Refine neighborhood-specific models

31. **Phase 3: Enhancement (Mar-Apr 2026)**
    - Add social media signals (Reddit, YouTube, Instagram)
    - Product-specific forecasts (tomatoes, salads, etc.)
    - Real-time forecasting dashboard
    - Competitor pricing integration

32. **Phase 4: Scale (Post-April 2026)**
    - Expand to other German cities (Munich, Hamburg)
    - Multi-product forecasting
    - Mobile app for AgriCom field teams
    - API integration with AgriCom systems

33. **Long-Term Vision**
    - Pan-European organic demand forecasting
    - Farmer supply coordination (upstream optimization)
    - Dynamic pricing based on demand forecast
    - Sustainability metrics (waste reduction tracking)

---

#### Closing (2 slides)

34. **Summary**
    - ✅ Built working forecasting model (82% accuracy)
    - ✅ 2-4 weeks lead time for inventory planning
    - ✅ Neighborhood-specific recommendations
    - ✅ Scalable, replicable methodology

35. **Thank You + Q&A**
    - Contact: Team names + emails
    - GitHub: github.com/matiascam02/ACP-AGRICOM
    - Open for questions

---

## Roles & Responsibilities

### Team Division

| Team | Members | Responsibilities | Presentation Slides |
|------|---------|------------------|---------------------|
| **Data Team** | TBD | Data collection, EDA, source justification | 11-14 (4 slides) |
| **Dev Team** | TBD | Modeling, forecasting, technical architecture | 15-19 (5 slides) |
| **Business Team** | TBD | Storyline, recommendations, usecase, roadmap | 20-27, 30-33 (11 slides) |
| **PM (Sean)** | Jean-Christophe | Opening, team intro, summary, Q&A moderation | 1-10, 28-29, 34-35 (14 slides) |

**Total:** 35 slides (distributed evenly, ~8-9 per person if 4 presenters)

---

### Presentation Flow (20-25 minutes)

| Time | Section | Presenter | Slides |
|------|---------|-----------|--------|
| 0-3 min | Opening + Problem | PM (Sean) | 1-6 |
| 3-6 min | Approach + Focus | PM (Sean) | 7-10 |
| 6-11 min | Data Team | Data Lead | 11-14 |
| 11-17 min | Dev Team | Dev Lead | 15-19 |
| 17-23 min | Business Value + Recs | Business Lead | 20-27 |
| 23-25 min | Roles + Roadmap | PM (Sean) | 28-33 |
| 25-27 min | Closing | PM (Sean) | 34-35 |
| 27-37 min | Q&A | All | - |

**Total:** 27-37 minutes (presentation + Q&A)

---

## Use Cases (Detailed)

### Use Case 1: Weekly Inventory Planning

**Scenario:**  
AgriCom needs to order organic produce from suppliers every Monday for the upcoming week.

**Without Our Model:**
- Order based on last week's sales
- Miss seasonal spikes (Christmas, holidays)
- Over-order during slow weeks
- Result: 20-25% waste rate

**With Our Model:**
- Check 2-week forecast every Friday
- Adjust order quantities based on prediction
- Prepare for events (match days, holidays)
- Result: 10-15% waste rate (estimated)

**Example:**
- **Week of Dec 18:** Forecast shows 2.5x spike
- **Action:** Order 150% more than usual
- **Outcome:** Capture Christmas demand, minimal stockouts

---

### Use Case 2: Promotional Campaign Timing

**Scenario:**  
AgriCom wants to run a "Fresh Salad Week" promotion.

**Without Our Model:**
- Schedule promotion randomly
- Risk: Low demand week = wasted marketing spend
- Risk: High demand week = stockouts

**With Our Model:**
- Identify warm, sunny weekend in forecast
- Schedule promotion for that week
- Ensure inventory matches expected demand
- Result: 30% higher campaign ROI (estimated)

**Example:**
- **Forecast:** Week of June 15 = 58.2 demand index (high)
- **Weather:** 22°C, sunny
- **Action:** Launch "Summer Salad" promotion
- **Outcome:** 40% sales lift vs. baseline

---

### Use Case 3: Geographic Expansion

**Scenario:**  
AgriCom wants to open a new organic market in Berlin.

**Without Our Model:**
- Choose location based on rent/foot traffic
- Risk: Low organic affinity neighborhood
- Risk: Cannibalize existing stores

**With Our Model:**
- Identify high-affinity neighborhood (Kreuzberg: 85%)
- Verify demand with Google Trends + sentiment
- Forecast potential sales volume
- Result: Data-driven site selection

**Example:**
- **Analysis:** Kreuzberg shows 85% affinity vs. 60% Charlottenburg
- **Action:** Open pilot store in Kreuzberg
- **Outcome:** 25% higher sales per sq meter (estimated)

---

### Use Case 4: Supplier Negotiation

**Scenario:**  
AgriCom negotiates contracts with organic farmers.

**Without Our Model:**
- Commit to fixed volumes quarterly
- Risk: Over-commit = waste
- Risk: Under-commit = stockouts

**With Our Model:**
- Share demand forecast with suppliers
- Negotiate flexible volume commitments
- Align farmer planting schedules with demand
- Result: Better supplier relationships + reduced waste

**Example:**
- **Forecast:** Q2 2026 avg demand = 52.3
- **Action:** Contract for 55 units/week (buffer)
- **Outcome:** 95% utilization vs. 70% before

---

### Use Case 5: Dynamic Pricing

**Scenario:**  
AgriCom wants to maximize revenue during demand fluctuations.

**Without Our Model:**
- Fixed pricing year-round
- Miss opportunity to charge premium during high demand
- Can't clear inventory during low demand

**With Our Model:**
- Price slightly higher during forecasted high-demand weeks
- Discount during forecasted low-demand weeks
- Optimize revenue vs. waste trade-off
- Result: 10-15% revenue lift (estimated)

**Example:**
- **Forecast:** Easter week demand = 59.2 (high)
- **Action:** +10% price premium on organic produce
- **Outcome:** Same sales volume, higher margin

---

## Future Roadmap (Detailed)

### Phase 2: Validation (Feb-Mar 2026)

**Objectives:**
- Complete data collection
- Validate model on real sales data
- Refine neighborhood models

**Key Activities:**
1. ✅ Complete Google Trends collection (all 20 keywords)
2. Run unified data merge with full dataset
3. Retrain model (expect R² improvement from 0.82 → 0.85+)
4. Request sample AgriCom sales data for validation
5. Build neighborhood-specific forecast models
6. Create automated weekly forecast report

**Deliverables:**
- Updated model with R² >0.85
- Validation report (model vs. actual sales)
- 3 neighborhood-specific models
- Weekly forecast automation script

**Timeline:** 4 weeks (Feb 10 - Mar 10)

---

### Phase 3: Enhancement (Mar-Apr 2026)

**Objectives:**
- Add new data sources
- Build product-specific forecasts
- Create dashboard for AgriCom

**Key Activities:**
1. Social media signals:
   - Reddit organic food discussions
   - YouTube recipe trends
   - Instagram influencer content
2. Product-level forecasting:
   - Tomatoes, salads, berries, etc.
   - Separate models per category
3. Real-time dashboard:
   - Live forecast updates
   - Interactive visualizations
   - Downloadable reports
4. Competitor analysis:
   - Track Rewe Bio, Edeka Bio pricing
   - Market share trends

**Deliverables:**
- Social media data pipeline
- 5 product-specific forecast models
- Interactive dashboard (Streamlit/Dash)
- Competitor tracking report

**Timeline:** 4 weeks (Mar 10 - Apr 5)

---

### Phase 4: Scale (Post-April 2026)

**Objectives:**
- Expand to other cities
- Multi-product forecasting
- API integration with AgriCom systems

**Key Activities:**
1. **Geographic expansion:**
   - Munich, Hamburg, Frankfurt models
   - City-specific data collection
   - Comparative analysis
2. **Product portfolio:**
   - 20+ product categories
   - Cross-product demand correlations
   - Basket analysis
3. **System integration:**
   - REST API for forecasts
   - Integration with AgriCom ERP/inventory system
   - Mobile app for field teams
4. **Advanced features:**
   - Supplier coordination (upstream optimization)
   - Dynamic pricing engine
   - Sustainability metrics dashboard

**Deliverables:**
- 4 city models (Berlin, Munich, Hamburg, Frankfurt)
- 20+ product forecasts
- Production-ready API
- Mobile app (iOS/Android)

**Timeline:** 3-6 months (Apr-Sep 2026)

---

### Long-Term Vision (2027+)

**Pan-European Organic Market Intelligence**

1. **Coverage:**
   - 20+ European cities
   - 100+ product categories
   - 1000+ suppliers integrated

2. **Features:**
   - Real-time demand forecasting
   - Farmer supply coordination
   - Dynamic pricing optimization
   - Waste reduction tracking (sustainability KPIs)
   - Market trend analysis (emerging products)

3. **Business Model:**
   - SaaS platform for organic retailers
   - Subscription: €500-5000/month (based on scale)
   - Farmer coordination marketplace (commission-based)

4. **Impact:**
   - 30-50% waste reduction across organic supply chain
   - 15-20% revenue lift for retailers
   - Better income stability for farmers
   - Environmental impact (reduced food waste)

---

## Key Messages for Business Team

### For AgriCom

✅ **Immediate Value:**  
2-4 weeks lead time for inventory decisions → 15-20% waste reduction

✅ **Competitive Advantage:**  
Predict demand before competitors (who rely on historical sales)

✅ **Scalable Solution:**  
Start with Berlin, expand to any city/product

✅ **Low Risk:**  
Pilot program = proof of concept before full rollout

✅ **Data-Driven:**  
Move from gut feeling to quantified forecasts

---

### For ESMT/Evaluators

✅ **Business Acumen:**  
Clear ROI estimation, pilot program design, phased rollout

✅ **Real-World Application:**  
Solves actual AgriCom pain points (waste, stockouts)

✅ **Strategic Thinking:**  
Neighborhood segmentation, long-term vision, scalability

✅ **Stakeholder Management:**  
Tailored recommendations for different audiences (AgriCom, farmers, consumers)

✅ **Execution Plan:**  
Detailed roadmap with timelines, deliverables, metrics

---

## Presentation Tips

### Storytelling Do's

✅ **Start with a hook:**  
"Did you know 25% of organic produce gets wasted before it reaches consumers?"

✅ **Use concrete examples:**  
"In December 2025, our model predicted a 2.8x spike. AgriCom would have ordered 150% more inventory."

✅ **Show, don't tell:**  
Use visualizations (forecast chart, neighborhood map, feature importance)

✅ **Connect to AgriCom's goals:**  
Every slide should tie back to business value

✅ **End with a call to action:**  
"Let's pilot this in Kreuzberg starting next month."

---

### Storytelling Don'ts

❌ **Avoid jargon:**  
Not "R² of 0.82" → "Explains 82% of demand variation"

❌ **Don't bury the lead:**  
Lead with results (82% accuracy), then explain how

❌ **Don't overcomplicate:**  
Business team doesn't need Ridge vs. RF technical details

❌ **Don't ignore limitations:**  
Acknowledge Google Trends incompleteness, offer mitigation

❌ **Don't oversell:**  
"15-20% waste reduction (estimated)" not "guaranteed"

---

## Q&A Preparation

### Expected Questions & Answers

**Q: How accurate is the model without complete Google Trends data?**  
A: Current R² = 0.82 with 8/20 keywords. We expect R² >0.85 once complete (similar models in literature achieve 0.80-0.90).

**Q: Can this work without actual AgriCom sales data?**  
A: Yes, we use Google Trends as a demand proxy (validated by Christmas spike, event patterns). Once AgriCom shares sales data, we can validate and further improve.

**Q: What if weather forecasts are wrong?**  
A: Weather is only 8.8% of model importance. Even with weather forecast errors, day-of-week (24.1%) and seasonality (21.5%) still provide strong signal.

**Q: How much would this cost to implement?**  
A: Data sources are free (Open-Meteo, GDELT, Google Trends). Compute: ~€50/month (AWS). Developer time: 1-2 days/week for monitoring. Total: <€500/month operational cost.

**Q: Why Kreuzberg for the pilot?**  
A: Highest organic affinity (85%), strong farmers market presence, weather-sensitive shoppers = ideal test case. Success here scales to other neighborhoods.

**Q: How often do you retrain the model?**  
A: Weekly with new data (automated). Major retraining quarterly to capture new patterns. Model stays current.

**Q: What about competitor actions (Rewe Bio sales, etc.)?**  
A: Phase 3 roadmap includes competitor pricing tracking. For now, model captures aggregate market demand (which competitors also respond to).

**Q: Can this predict product-specific demand (just tomatoes)?**  
A: Phase 3 deliverable. Current model = aggregate organic demand. Product-specific models follow same methodology (just different Google Trends keywords).

---

## Next Steps for Business Team

### This Week
1. ✅ Finalize presentation structure (this document)
2. ⏳ Create slide deck (use AGRICOM_PRESENTATION_SLIDES.md as base)
3. ⏳ Coordinate with Data/Dev teams on slide content
4. ⏳ Prepare speaker notes

### Presentation Prep
1. Rehearse timing (aim for 20-22 min, leave buffer for Q&A)
2. Assign slides to presenters (4 people = ~8 slides each)
3. Prepare Q&A responses (brainstorm 10-15 likely questions)
4. Create backup slides (technical details, if asked)

### Post-Presentation
1. Gather feedback from AgriCom/mentor
2. Update roadmap based on input
3. Begin Phase 2 activities (complete Trends data)
4. Schedule pilot program kickoff (if approved)

---

**Contact:** Business Team Lead  
**Presentation Date:** TBD (Week of Feb 17?)  
**Audience:** AgriCom (Farah Ezzedine), Mentor (Ryan Krog), ESMT Faculty
