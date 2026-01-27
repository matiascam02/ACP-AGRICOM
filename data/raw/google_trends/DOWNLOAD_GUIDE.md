# Google Trends Manual Download Guide
**Generated:** 2026-01-27

## Why Manual Download?
Google Trends blocks automated collection (rate limits). Each team member should download 2-3 keywords manually. This takes ~5 minutes per person.

---

## Instructions

1. Go to https://trends.google.com/trends/
2. Enter the keyword in the search box
3. Set filters:
   - **Location:** Germany (or Berlin if specified)
   - **Time range:** Past 5 years
   - **Category:** All categories
4. Click the **download button** (↓ arrow icon, top-right of chart)
5. Save the CSV file with the format shown below
6. Upload to: `data/raw/google_trends/` in the GitHub repo

---

## Keywords to Download

### Already Collected ✅
- ~~bio gemüse~~ ✅
- ~~bio tomaten~~ ✅
- ~~bio company berlin~~ ✅
- ~~bio lebensmittel berlin~~ ✅
- ~~lidl bio~~ ✅
- ~~suppe rezepte~~ ✅

### HIGH PRIORITY 🔴

| Keyword | Location | Save As |
|---------|----------|---------|
| bio lebensmittel | Germany | `trends_bio_lebensmittel_de_5y.csv` |
| wochenmarkt berlin | Germany | `trends_wochenmarkt_berlin_de_5y.csv` |
| bio obst | Germany | `trends_bio_obst_de_5y.csv` |
| bio supermarkt | Germany | `trends_bio_supermarkt_de_5y.csv` |
| bauernmarkt | Germany | `trends_bauernmarkt_de_5y.csv` |
| alnatura | Germany | `trends_alnatura_de_5y.csv` |

### MEDIUM PRIORITY 🟡

| Keyword | Location | Save As |
|---------|----------|---------|
| rewe bio | Germany | `trends_rewe_bio_de_5y.csv` |
| edeka bio | Germany | `trends_edeka_bio_de_5y.csv` |
| nachhaltig einkaufen | Germany | `trends_nachhaltig_einkaufen_de_5y.csv` |
| vegan lebensmittel | Germany | `trends_vegan_lebensmittel_de_5y.csv` |
| regional einkaufen | Germany | `trends_regional_einkaufen_de_5y.csv` |
| grillen rezepte | Germany | `trends_grillen_rezepte_de_5y.csv` |
| salat rezepte | Germany | `trends_salat_rezepte_de_5y.csv` |

### LOWER PRIORITY 🟢

| Keyword | Location | Save As |
|---------|----------|---------|
| bio lieferung | Germany | `trends_bio_lieferung_de_5y.csv` |
| smoothie rezepte | Germany | `trends_smoothie_rezepte_de_5y.csv` |
| zero waste | Germany | `trends_zero_waste_de_5y.csv` |

---

## Team Assignments

| Team Member | Keywords to Download |
|-------------|---------------------|
| Member 1 | bio lebensmittel, wochenmarkt berlin, alnatura |
| Member 2 | bio obst, bauernmarkt, rewe bio |
| Member 3 | bio supermarkt, edeka bio, nachhaltig einkaufen |
| Member 4 | vegan lebensmittel, regional einkaufen, grillen rezepte |
| Member 5 | salat rezepte, bio lieferung, smoothie rezepte |
| Member 6 | zero waste (+ any missing from others) |

---

## Upload Instructions

1. Clone or pull the repo: `git clone https://github.com/matiascam02/ACP-AGRICOM.git`
2. Add your CSV files to `data/raw/google_trends/`
3. Commit and push:
   ```bash
   git add data/raw/google_trends/
   git commit -m "Add Google Trends data: [your keywords]"
   git push
   ```

Or upload directly via GitHub web interface.

---

## Questions?
Contact Matias or check SESSION_STATUS.md
