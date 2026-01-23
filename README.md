# AGRICOM Project - Organic Produce Demand Forecasting

## Project Overview
Developing a methodology to predict organic produce demand in Berlin using alternative/external data signals (Google Trends, weather, events, social media) as leading indicators.

**Client:** AgriCom
**Timeline:** January - March 2026
**Team:** ESMT MBA Students

## Project Structure

```
agricom_project/
├── data/
│   ├── raw/              # Original data files
│   ├── processed/        # Cleaned and transformed data
│   └── external/         # Third-party data sources
├── notebooks/            # Jupyter notebooks for analysis
├── src/
│   ├── data_collection/  # Scripts to fetch data from APIs
│   ├── analysis/         # Statistical analysis modules
│   └── visualization/    # Plotting and dashboard code
├── outputs/
│   ├── figures/          # Generated charts and graphs
│   └── reports/          # Analysis reports and summaries
└── README.md
```

## Data Sources

| Source | API/Method | Status |
|--------|------------|--------|
| Google Trends | pytrends | Ready |
| Weather | Open-Meteo API | Ready |
| Events | Manual + APIs | Pending |
| News | GDELT/NewsAPI | Pending |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Collect Google Trends data
python src/data_collection/google_trends.py

# Collect weather data
python src/data_collection/weather.py

# Run analysis
python src/analysis/correlation_analysis.py
```

## Key Hypotheses

1. Google Trends predicts demand 2-4 weeks ahead
2. Weather anomalies drive demand shifts within 3-7 days
3. Football matches create localized demand patterns
4. Chef influencer content predicts ingredient spikes
5. Vacation returns predict tomato demand
6. News sentiment affects organic premium willingness

## Documentation

Full documentation available in Obsidian:
`40_Projects/ESMT/AGRICOM/AGRICOM/claudes-work/`

## Team

- Project Manager: Jean-Christophe (Sean)
- Students: Matias, Ryan/Tonghan, Arjun, Akos, May
- Mentor: Ryan Krog (Deutsche Bank)
- Client Contact: Farah Ezzedine (AgriCom)
