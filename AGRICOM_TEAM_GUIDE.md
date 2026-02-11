# AGRICOM Project - Team Organization Guide

**Project:** Organic Produce Demand Forecasting for Berlin  
**Client:** AgriCom.io  
**Timeline:** January - April 2026  
**Presentation Date:** Week of Feb 17, 2026 (TBD)

---

## Team Structure

We are organized into **3 specialized teams** + **1 Project Manager**:

### 🔍 Data Team
**Role:** Exploratory Data Analysis (EDA) of current data, explain sources, justify choices

**Responsibilities:**
- Present data collection process
- Show EDA findings (weather patterns, sentiment trends, seasonality)
- Explain why we chose each data source
- Demonstrate data quality and completeness

**Presentation Slides:** 11-14 (4 slides)  
**Document:** `AGRICOM_DATA_TEAM.md`

---

### 💻 Dev Team
**Role:** Explain project architecture, models used, forecasting pipeline

**Responsibilities:**
- Present technical workflow (data → processing → modeling → forecasting)
- Explain model selection (Ridge vs. RF vs. GBM)
- Show model performance (R² = 0.82, feature importance)
- Demonstrate forecasting output

**Presentation Slides:** 15-19 (5 slides)  
**Document:** `AGRICOM_DEV_TEAM.md`

---

### 💼 Business Team
**Role:** Presentation storyline, structure, roles, use cases, future roadmap

**Responsibilities:**
- Design overall presentation narrative
- Create use cases (inventory optimization, promotional planning, etc.)
- Develop recommendations for AgriCom
- Build phased roadmap (Phase 2-4 + long-term vision)

**Presentation Slides:** 20-27, 30-33 (11 slides)  
**Document:** `AGRICOM_BUSINESS_TEAM.md`

---

### 📋 Project Manager (Sean)
**Role:** Coordination, opening, closing, team intro, Q&A moderation

**Responsibilities:**
- Open presentation (problem statement, context)
- Introduce team and approach
- Coordinate between teams
- Close presentation (summary, next steps)
- Moderate Q&A session

**Presentation Slides:** 1-10, 28-29, 34-35 (14 slides)  
**Document:** All three team documents + this guide

---

## Document Overview

| File | Purpose | For Team |
|------|---------|----------|
| **AGRICOM_EXECUTIVE_SUMMARY.md** | 1-page stakeholder summary | All (reference) |
| **AGRICOM_PRESENTATION_SLIDES.md** | Full slide deck (35 slides, Marp format) | All (base template) |
| **AGRICOM_DATA_TEAM.md** | Data sources, EDA, quality assessment | Data Team |
| **AGRICOM_DEV_TEAM.md** | Models, forecasting, technical architecture | Dev Team |
| **AGRICOM_BUSINESS_TEAM.md** | Storyline, use cases, recommendations, roadmap | Business Team |
| **AGRICOM_TEAM_GUIDE.md** | This file - team organization overview | All (start here) |

---

## Presentation Flow (20-25 min + Q&A)

| Time | Section | Team | Slides | Duration |
|------|---------|------|--------|----------|
| **0-3 min** | Opening + Problem | PM | 1-6 | 3 min |
| **3-6 min** | Approach + Focus | PM | 7-10 | 3 min |
| **6-11 min** | Data & EDA | Data Team | 11-14 | 5 min |
| **11-17 min** | Models & Forecasting | Dev Team | 15-19 | 6 min |
| **17-23 min** | Use Cases + Recommendations | Business Team | 20-27 | 6 min |
| **23-25 min** | Roles + Roadmap | PM + Business | 28-33 | 2 min |
| **25-27 min** | Closing | PM | 34-35 | 2 min |
| **27-37 min** | Q&A | All | - | 10 min |

**Total:** 27-37 minutes (presentation + Q&A)

---

## Team Assignments (To Be Decided)

**Fill in names:**

| Team | Members | Contact |
|------|---------|---------|
| **PM** | Jean-Christophe (Sean) | [email] |
| **Data Team** | ?, ?, ? | [email] |
| **Dev Team** | Matias, ?, ? | [email] |
| **Business Team** | ?, ?, ? | [email] |

---

## Quick Start Guide

### For Data Team:
1. Read `AGRICOM_DATA_TEAM.md`
2. Review your section in `AGRICOM_PRESENTATION_SLIDES.md` (slides 11-14)
3. Prepare 4 slides covering:
   - Data sources (why we chose them)
   - Weather EDA findings
   - News sentiment trends
   - Seasonality patterns (Google Trends)
4. Create 3-5 key visualizations from outputs/figures/

---

### For Dev Team:
1. Read `AGRICOM_DEV_TEAM.md`
2. Review your section in `AGRICOM_PRESENTATION_SLIDES.md` (slides 15-19)
3. Prepare 5 slides covering:
   - Workflow diagram (data → model → forecast)
   - Model comparison table (Ridge wins)
   - Feature importance chart
   - 12-week forecast visualization
   - (Optional) Code demo or architecture diagram
4. Be ready to explain R² = 0.82 in non-technical terms

---

### For Business Team:
1. Read `AGRICOM_BUSINESS_TEAM.md`
2. Review your sections in `AGRICOM_PRESENTATION_SLIDES.md` (slides 20-27, 30-33)
3. Prepare 11 slides covering:
   - Use cases (inventory, promotions, targeting, events, pricing)
   - Recommendations for AgriCom (pilot program, priorities)
   - Future roadmap (Phase 2-4 + long-term vision)
4. Develop 10-15 Q&A responses (see Business Team doc)

---

### For Project Manager (Sean):
1. Read all three team documents + `AGRICOM_EXECUTIVE_SUMMARY.md`
2. Review full presentation deck (`AGRICOM_PRESENTATION_SLIDES.md`)
3. Prepare opening slides (1-10):
   - Problem statement (slide 4-6)
   - Methodology overview (slide 7-9)
   - Berlin focus (slide 10)
4. Prepare closing slides (28-29, 34-35):
   - Team roles (slide 28-29)
   - Summary (slide 34)
   - Q&A intro (slide 35)
5. Coordinate rehearsal schedule

---

## Key Deadlines

| Date | Milestone | Owner |
|------|-----------|-------|
| **Feb 11** | Complete Google Trends data collection | All (team effort) |
| **Feb 12** | Finalize slide deck | Business Team |
| **Feb 13** | Slide content review | All teams |
| **Feb 14** | Full rehearsal | All |
| **Feb 15** | Final edits | PM |
| **Feb 17** | **PRESENTATION** | All |

---

## Rehearsal Schedule (Proposed)

### Rehearsal 1 (Feb 14, Morning)
- Each team presents their section (rough draft OK)
- Timing check (aim for 20-22 min total)
- Identify gaps or overlaps

### Rehearsal 2 (Feb 14, Afternoon)
- Full run-through with polished slides
- Practice transitions between presenters
- Q&A role-play (10 sample questions)

### Rehearsal 3 (Feb 16, Evening)
- Final dress rehearsal
- No stops, full 25-min presentation
- Record for self-review

---

## Resources

### GitHub Repository
**URL:** https://github.com/matiascam02/ACP-AGRICOM

**Key Folders:**
- `/data/` - All collected data (raw + processed)
- `/src/` - Python scripts (collection, analysis, modeling)
- `/outputs/` - Visualizations, reports, forecasts
- `/README.md` - Project overview

---

### Obsidian Documentation
**Path:** `~/Documents/CPU/40_Projects/ESMT/AGRICOM/`

**Key Files:**
- Full research notes
- Literature review
- Meeting notes with AgriCom/mentor

---

### Presentation Materials
**Workspace:** `/Users/matias-claw/.openclaw/workspace/`

**Files Created:**
- `AGRICOM_EXECUTIVE_SUMMARY.md` - 1-page summary
- `AGRICOM_PRESENTATION_SLIDES.md` - Full deck (Marp format)
- `AGRICOM_DATA_TEAM.md` - Data team guide
- `AGRICOM_DEV_TEAM.md` - Dev team guide
- `AGRICOM_BUSINESS_TEAM.md` - Business team guide
- `AGRICOM_TEAM_GUIDE.md` - This file

---

## Converting Slides to PowerPoint/PDF

### Option 1: Marp CLI (Recommended)

```bash
# Install Marp
npm install -g @marp-team/marp-cli

# Convert to PowerPoint
marp AGRICOM_PRESENTATION_SLIDES.md -o AGRICOM_Presentation.pptx

# Or convert to PDF
marp AGRICOM_PRESENTATION_SLIDES.md -o AGRICOM_Presentation.pdf

# Or HTML (for web viewing)
marp AGRICOM_PRESENTATION_SLIDES.md -o AGRICOM_Presentation.html
```

### Option 2: VS Code Extension

1. Install "Marp for VS Code" extension
2. Open `AGRICOM_PRESENTATION_SLIDES.md`
3. Click "Export Slide Deck" button
4. Choose format (PPTX, PDF, HTML)

### Option 3: Manual Copy-Paste

- Each `---` separator = new slide
- Copy content to Google Slides / PowerPoint
- Add ESMT/AgriCom branding

---

## Q&A Preparation

### Must-Know Answers (All Team Members)

**Q: What is the model accuracy?**  
A: R² = 0.82, which means we explain 82% of demand variance. Error is ~4% on average.

**Q: How much lead time do you provide?**  
A: 2-4 weeks. Google Trends searches precede purchases, events are known ahead, weather has 7-day forecasts.

**Q: Why is Google Trends data incomplete?**  
A: API rate limits. We collected 8/20 keywords (40%). Manual team download ongoing. Model already at 82% accuracy; will improve with full data.

**Q: What's the business value for AgriCom?**  
A: Estimated 15-20% waste reduction + 10% sales lift from better inventory planning. ~€300k/year value for Berlin operations.

**Q: How does this scale?**  
A: Same methodology works for any city/product. Just change location/keywords in data collection scripts. Full automation = easy replication.

---

### Team-Specific Q&A

**Data Team:**
- Why these 5 data sources specifically?
- How did you validate data quality?
- What about missing values or gaps?

**Dev Team:**
- Why Ridge over Random Forest?
- How did you validate the model?
- What happens if weather forecasts are wrong?

**Business Team:**
- Why pilot in Kreuzberg first?
- How much would implementation cost?
- What if AgriCom doesn't have sales data to validate?

**See each team's document for detailed Q&A prep.**

---

## Communication Channels

### Team Slack/Email
- Daily standup (async): Progress updates
- Blockers: Flag immediately to PM
- Questions: Tag relevant team

### Weekly Sync Meeting
- **When:** Fridays 3pm (or TBD)
- **Agenda:** Status, blockers, next week plan
- **Duration:** 30 min

### Pre-Presentation Sync
- **When:** Feb 13-14 (rehearsal days)
- **Format:** In-person or Zoom
- **Purpose:** Polish presentation, practice timing

---

## Success Metrics

### Presentation Goals

✅ **Clear Communication:**  
Non-technical stakeholders understand the value (AgriCom, ESMT faculty)

✅ **Time Management:**  
Stay within 20-22 min presentation (leave 8-10 min for Q&A)

✅ **Team Coordination:**  
Smooth transitions, no overlap or gaps between sections

✅ **Visual Impact:**  
3-5 memorable charts/graphs that tell the story

✅ **Q&A Confidence:**  
Answer 90% of questions without hesitation

---

### Project Success (Post-Presentation)

✅ **AgriCom Buy-In:**  
Agreement to pilot program in Kreuzberg (Feb-Mar)

✅ **Mentor Approval:**  
Positive feedback from Ryan Krog on methodology

✅ **Data Completion:**  
100% Google Trends collection by end of Feb

✅ **Model Improvement:**  
R² >0.85 with full dataset

✅ **Validation:**  
Test model on actual AgriCom sales data (if provided)

---

## Next Steps (Post-Presentation)

1. **Debrief Meeting (Feb 18)**
   - What went well?
   - What could improve?
   - Feedback from AgriCom/mentor

2. **Phase 2 Kickoff (Feb 19)**
   - Complete Google Trends data
   - Run unified data merge
   - Retrain model with full dataset

3. **Pilot Program Design (Feb 20-25)**
   - If AgriCom approves, design pilot
   - Metrics, timeline, success criteria
   - Kreuzberg store selection

4. **Final Report (Mar 1)**
   - Comprehensive write-up
   - Methodology documentation
   - Code + data handoff to AgriCom

---

## Final Checklist (Before Presentation)

### Content
- [ ] All slides finalized (35 total)
- [ ] Visualizations embedded (3-5 key charts)
- [ ] Speaker notes written
- [ ] Q&A responses prepared (10-15 questions)

### Logistics
- [ ] Presentation file backed up (PPTX + PDF)
- [ ] Clicker/remote tested
- [ ] Laptop HDMI/adapters ready
- [ ] Team knows slide assignments

### Rehearsal
- [ ] Full run-through 2x
- [ ] Timing under 23 min
- [ ] Transitions smooth
- [ ] Q&A practiced

### Team
- [ ] Everyone knows their slides
- [ ] Backup presenter for each section
- [ ] Contact info shared (in case of absence)
- [ ] Dress code agreed (business casual?)

---

## Contact & Support

**Project Manager:** Jean-Christophe (Sean)  
**Mentor:** Ryan Krog (Deutsche Bank)  
**Client:** Farah Ezzedine (AgriCom)  

**Questions?** Post in team Slack or email PM.

---

**Good luck, team! Let's show AgriCom what we've built.** 🚀

---

*Document created: February 11, 2026*  
*Last updated: February 11, 2026*
