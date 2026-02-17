# Data Used

This document describes all data sources used in the AGRICOM project, including their origin, collection method, temporal coverage, geographic scope, and variables.

---

## 1. Weather Data

**Source:** Open-Meteo API (free, no API key required)

- Archive API: `https://archive-api.open-meteo.com/v1/archive`
- Forecast API: `https://api.open-meteo.com/v1/forecast`

**Collection script:** `src/data_collection/weather.py`

**Geographic scope:** Berlin, Germany (latitude 52.52, longitude 13.405)

**Temporal coverage:** ~3 years of historical daily data (ending ~1 week before collection date) plus a 14-day weather forecast.

**Frequency:** Daily

**Output file:** `data/raw/weather_berlin_YYYYMMDD.csv`

### Variables

| Variable | Type | Description |
|---|---|---|
| `temperature_2m_max` | float | Daily maximum temperature at 2m height (°C) |
| `temperature_2m_min` | float | Daily minimum temperature at 2m height (°C) |
| `temperature_2m_mean` | float | Daily mean temperature at 2m height (°C) |
| `precipitation_sum` | float | Total daily precipitation (mm) |
| `rain_sum` | float | Total daily rain (mm) |
| `snowfall_sum` | float | Total daily snowfall (cm) |
| `precipitation_hours` | float | Number of hours with precipitation |
| `wind_speed_10m_max` | float | Maximum wind speed at 10m height (km/h) |
| `wind_gusts_10m_max` | float | Maximum wind gusts at 10m height (km/h) |
| `sunshine_duration` | float | Total sunshine duration (seconds) |
| `et0_fao_evapotranspiration` | float | FAO reference evapotranspiration (mm) |
| `is_forecast` | bool | Whether the row is a forecast (True) or historical observation (False) |

---

## 2. Events Data

**Sources:**
- **Bundesliga matches:** OpenLigaDB API (`https://api.openligadb.de/getmatchdata/bl1`) -- free, no API key required
- **School holidays, public holidays, major events:** Hardcoded reference data within the script based on official German calendar data

**Collection script:** `src/data_collection/events.py`

**Geographic scope:** Berlin, Germany

**Temporal coverage:** 2023--2026 (Bundesliga seasons 2023/24, 2024/25, 2025/26; holidays and events from 2024--2026)

**Frequency:** Event-based (each row is a single event or event-day)

**Output file:** `data/raw/events_berlin_YYYYMMDD.csv`

### Variables

| Variable | Type | Description |
|---|---|---|
| `date` | datetime | Date of the event |
| `event_type` | string | Category: `football_home`, `holiday_start`, `holiday_end`, `public_holiday`, `trade_fair`, `sports`, `festival`, `market` |
| `event_name` | string | Name of the event (e.g., "Union Berlin vs Bayern Munich", "Easter 2025 starts", "Christmas Markets") |
| `team` | string | Berlin football team (Hertha BSC or Union Berlin) -- football events only |
| `stadium` | string | Stadium name -- football events only |
| `affected_neighborhoods` | string | Neighborhoods expected to be impacted (e.g., "Charlottenburg, Westend" or "All") |
| `expected_attendance` | float | Estimated stadium attendance (capacity x 0.8) -- football events only |
| `holiday_duration` | int | Total duration of the school holiday in days -- holiday events only |
| `expected_impact` | string | Expected demand impact category (`reduced_demand`, `vacation_return_effect`) -- holidays only |
| `stores_closed` | bool | Whether stores are expected to be closed -- public holidays only |
| `day_of_event` | int | Current day number within a multi-day event |
| `total_days` | int | Total number of days for a multi-day event |

### Event Types Covered

- **Bundesliga home matches:** Hertha BSC (Olympiastadion, capacity 74,475) and Union Berlin (Stadion An der Alten Forsterei, capacity 22,012)
- **School holidays:** Winter, Easter, Summer, Autumn, Christmas breaks for Berlin (2024--2026)
- **Public holidays:** German national holidays (New Year, Good Friday, Easter Monday, Labour Day, Ascension Day, Whit Monday, German Unity Day, Christmas Day, Boxing Day)
- **Major Berlin events:** Fruit Logistica (trade fair), Berlin Marathon, Karneval der Kulturen, Berlin Festival of Lights, Christmas Markets

---

## 3. GDELT News Sentiment Data

**Source:** GDELT DOC API via the `gdeltdoc` Python package

**Collection script:** `src/data_collection/gdelt_news.py`

**Geographic scope:** Germany (country code `GM`)

**Temporal coverage:** Last 2 years from the date of collection

**Frequency:** Sub-daily (timeline tone), per-article (articles)

**Output files:**
- `data/raw/gdelt_articles_YYYYMMDD.csv` -- individual news articles
- `data/raw/gdelt_timeline_YYYYMMDD.csv` -- daily tone/sentiment timeline

### Search Queries

| Query Name | Keywords | Description |
|---|---|---|
| `organic_food` | bio lebensmittel, organic food, okologisch | General organic food coverage |
| `organic_produce` | bio gemuse, bio obst, organic vegetables | Organic produce coverage |
| `food_safety` | lebensmittelskandal, pestizide lebensmittel, food safety germany | Food safety and scandal coverage |
| `sustainable_food` | nachhaltige ernahrung, sustainable food, klimafreundlich essen | Sustainable food coverage |
| `farmers_markets` | wochenmarkt, bauernmarkt, farmers market germany | Farmers market coverage |
| `plant_based` | vegan deutschland, vegetarisch trend, plant based germany | Plant-based food trend coverage |

### Articles Variables

| Variable | Type | Description |
|---|---|---|
| `title` | string | Article headline |
| `url` | string | Full article URL |
| `domain` | string | Publishing domain |
| `language` | string | Article language |
| `seendate` | datetime | Date the article was observed by GDELT |
| `tone` | float | Article tone/sentiment score |
| `query_name` | string | Which search query matched this article |

### Timeline Variables

| Variable | Type | Description |
|---|---|---|
| `datetime` | datetime | Timestamp of the data point |
| `Average Tone` | float | Average sentiment tone score (positive = positive coverage, negative = negative coverage) |
| `query_name` | string | Which search query this tone relates to |

---

## 4. Google Trends Data

**Source:** Google Trends (via multiple collection methods)
- `pytrends` Python library (`src/data_collection/google_trends.py`, `google_trends_simple.py`, `google_trends_chunked.py`)
- SerpAPI (`src/data_collection/google_trends_serpapi.py`)
- Manual CSV download from Google Trends web interface

**Geographic scope:** Germany-wide (`DE`) -- Berlin-specific (`DE-BE`) was attempted but yielded sparse data for organic food terms

**Temporal coverage:** 5 years (approximately 2021--2026)

**Frequency:** Weekly

**Output files:** Individual CSV files per keyword in `data/raw/google_trends/`, named as `trends_<keyword>_de_5y.csv`

### Keywords Tracked

| Category | Keywords | Purpose |
|---|---|---|
| **Organic food** | bio lebensmittel, bio gemuse, bio obst, bio tomaten, organic food | Core organic food search interest |
| **Shopping behavior** | wochenmarkt berlin, bauernmarkt, bio supermarkt | Behavioral proxy for shopping intent |
| **Retailers** | alnatura, bio company berlin, rewe bio, edeka bio, lidl bio | Demand proxies via retailer interest |
| **Seasonal/recipe signals** | grillen rezepte, salat rezepte, suppe rezepte, smoothie rezepte | Seasonal cooking behavior |
| **Sustainability/lifestyle** | nachhaltig einkaufen, vegan lebensmittel, regional einkaufen, zero waste | Sustainability lifestyle interest |

### Variables

| Variable | Type | Description |
|---|---|---|
| `date` / `Week` | datetime | Start date of the week |
| `value` / keyword column | int (0--100) | Relative search interest (100 = peak popularity for that term in the given timeframe and geography) |
| `keyword` | string | The search term |

### Collection Status

At the time of the last collection, 8 out of 20 target keywords had been successfully collected. Collection is tracked in `data/raw/google_trends/progress.json`.

---

## 5. Economic Indicators

**Sources:**
- **OECD SDMX API:** Consumer Confidence Index (CCI) for Germany
- **World Bank API:** Inflation rate (`FP.CPI.TOTL.ZG`), GDP growth (`NY.GDP.MKTP.KD.ZG`)
- **Eurostat API:** Harmonized Index of Consumer Prices (HICP) for food and non-alcoholic beverages (dataset `prc_hicp_midx`, COICOP `CP01`, index 2015=100)
- **Bundesbank API:** ECB main refinancing interest rate

**Collection script:** `src/data_collection/economic_indicators.py`

**Geographic scope:** Germany (`DE` / `DEU`)

**Temporal coverage:** Varies by source; generally 2019--2026 for World Bank, longer for OECD/Eurostat

**Frequency:** Monthly

**Output file:** `data/raw/economic_indicators_YYYYMMDD.csv`

### Variables

| Variable | Type | Description |
|---|---|---|
| `date` | datetime | First day of the month |
| `consumer_confidence` | float | OECD Consumer Confidence Index (>100 = optimistic, <100 = pessimistic) |
| `inflation_rate` | float | Annual inflation rate (%) |
| `gdp_growth` | float | Annual GDP growth rate (%) |
| `food_price_index` | float | HICP food and non-alcoholic beverages index (2015=100) |
| `interest_rate` | float | ECB main refinancing rate (%) |

---

## 6. Social Signals

**Sources:**
- **YouTube Data API:** Search volume for organic food-related terms (requires `YOUTUBE_API_KEY`)
- **Reddit API:** Post and comment activity in German subreddits (requires `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`; uses PRAW library)
- **Instagram:** Hashtag popularity via RapidAPI (requires `RAPIDAPI_KEY`)

**Collection scripts:** `src/data_collection/social_signals.py`, `src/data_collection/reddit_collector.py`

**Geographic scope:** Germany / German-language content

**Temporal coverage:** Snapshot at collection time (YouTube: recent videos; Reddit: last month/year; Instagram: cumulative post counts)

**Frequency:** Point-in-time snapshot

**Output files:** `data/raw/social_signals_YYYYMMDD.csv`, `data/raw/reddit_posts_YYYYMMDD.csv`

### YouTube Variables

| Variable | Type | Description |
|---|---|---|
| `term` | string | Search term (e.g., "bio kochen", "vegane rezepte") |
| `youtube_recent_videos` | int | Count of recent videos matching the term |
| `youtube_total_results` | int | Total number of results for the term |

### Reddit Variables

| Variable | Type | Description |
|---|---|---|
| `subreddit` | string | Subreddit name (berlin, germany, de, Kochen, VeganDE, Finanzen) |
| `query` | string | Search query used |
| `post_id` | string | Unique post identifier |
| `title` | string | Post title |
| `selftext` | string | Post body text (first 500 characters) |
| `score` | int | Reddit score (upvotes minus downvotes) |
| `upvote_ratio` | float | Ratio of upvotes to total votes |
| `num_comments` | int | Number of comments on the post |
| `created_utc` | datetime | Post creation timestamp |

### Reddit Search Queries

bio lebensmittel, organic food, wochenmarkt, bauernmarkt, farmers market, nachhaltig essen, vegan berlin, supermarkt, REWE bio, Alnatura, Bio Company

### Instagram Variables

| Variable | Type | Description |
|---|---|---|
| `hashtag` | string | Tracked hashtag (e.g., biolebensmittel, biogemuse, wochenmarkt, bauernmarkt, bioberlin) |
| `post_count` | int | Total number of posts with this hashtag |

### Composite Social Interest Index

| Variable | Type | Description |
|---|---|---|
| `social_interest_index` | float (0--100) | Normalized composite index combining YouTube video count, Reddit post count, and Reddit engagement |

---

## Data Summary

| Data Source | API / Provider | Frequency | Coverage | Key Signal |
|---|---|---|---|---|
| Weather | Open-Meteo | Daily | ~3 years + 14-day forecast | Temperature, precipitation, sunshine |
| Events | OpenLigaDB + reference data | Event-based | 2023--2026 | Football matches, holidays, festivals |
| News Sentiment | GDELT | Sub-daily | Last 2 years | Media tone on organic food topics |
| Google Trends | Google Trends | Weekly | 5 years | Search interest for organic food terms |
| Economic Indicators | OECD, World Bank, Eurostat, Bundesbank | Monthly | 2019--2026 | Consumer confidence, inflation, food prices |
| Social Signals | YouTube, Reddit, Instagram | Snapshot | Point-in-time | Social media engagement with organic food topics |
