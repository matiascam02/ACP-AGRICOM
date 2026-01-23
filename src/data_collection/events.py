"""
AGRICOM - Events Data Collection
Collects Berlin events data: Bundesliga matches, holidays, major events.

Usage:
    python events.py

Output:
    data/raw/events_berlin_YYYYMMDD.csv
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')

# Berlin football teams and their stadiums
BERLIN_TEAMS = {
    'Hertha BSC': {
        'stadium': 'Olympiastadion',
        'lat': 52.5147,
        'lon': 13.2395,
        'capacity': 74475,
        'neighborhoods': ['Charlottenburg', 'Westend']
    },
    'Union Berlin': {
        'stadium': 'Stadion An der Alten Försterei',
        'lat': 52.4573,
        'lon': 13.5680,
        'capacity': 22012,
        'neighborhoods': ['Köpenick', 'Treptow']
    }
}

# German school holidays (Berlin 2024-2026)
SCHOOL_HOLIDAYS = [
    # 2024
    {'name': 'Winter 2024', 'start': '2024-02-05', 'end': '2024-02-10'},
    {'name': 'Easter 2024', 'start': '2024-03-25', 'end': '2024-04-05'},
    {'name': 'Summer 2024', 'start': '2024-07-18', 'end': '2024-08-30'},
    {'name': 'Autumn 2024', 'start': '2024-10-21', 'end': '2024-11-02'},
    {'name': 'Christmas 2024', 'start': '2024-12-23', 'end': '2025-01-03'},
    # 2025
    {'name': 'Winter 2025', 'start': '2025-02-03', 'end': '2025-02-08'},
    {'name': 'Easter 2025', 'start': '2025-04-14', 'end': '2025-04-25'},
    {'name': 'Summer 2025', 'start': '2025-07-24', 'end': '2025-09-06'},
    {'name': 'Autumn 2025', 'start': '2025-10-20', 'end': '2025-11-01'},
    {'name': 'Christmas 2025', 'start': '2025-12-22', 'end': '2026-01-03'},
    # 2026
    {'name': 'Winter 2026', 'start': '2026-02-02', 'end': '2026-02-07'},
    {'name': 'Easter 2026', 'start': '2026-03-30', 'end': '2026-04-11'},
]

# Major Berlin events
MAJOR_EVENTS = [
    # Annual recurring events
    {'name': 'Fruit Logistica', 'type': 'trade_fair', 'date': '2024-02-07', 'duration': 3},
    {'name': 'Fruit Logistica', 'type': 'trade_fair', 'date': '2025-02-05', 'duration': 3},
    {'name': 'Fruit Logistica', 'type': 'trade_fair', 'date': '2026-02-04', 'duration': 3},
    {'name': 'Berlin Marathon', 'type': 'sports', 'date': '2024-09-29', 'duration': 1},
    {'name': 'Berlin Marathon', 'type': 'sports', 'date': '2025-09-28', 'duration': 1},
    {'name': 'Karneval der Kulturen', 'type': 'festival', 'date': '2024-05-17', 'duration': 4},
    {'name': 'Karneval der Kulturen', 'type': 'festival', 'date': '2025-05-30', 'duration': 4},
    {'name': 'Berlin Festival of Lights', 'type': 'festival', 'date': '2024-10-04', 'duration': 10},
    {'name': 'Berlin Festival of Lights', 'type': 'festival', 'date': '2025-10-03', 'duration': 10},
    {'name': 'Christmas Markets', 'type': 'market', 'date': '2024-11-25', 'duration': 30},
    {'name': 'Christmas Markets', 'type': 'market', 'date': '2025-11-24', 'duration': 30},
]

# German public holidays
PUBLIC_HOLIDAYS = [
    # 2024
    {'name': 'New Year', 'date': '2024-01-01'},
    {'name': 'Good Friday', 'date': '2024-03-29'},
    {'name': 'Easter Monday', 'date': '2024-04-01'},
    {'name': 'Labour Day', 'date': '2024-05-01'},
    {'name': 'Ascension Day', 'date': '2024-05-09'},
    {'name': 'Whit Monday', 'date': '2024-05-20'},
    {'name': 'German Unity Day', 'date': '2024-10-03'},
    {'name': 'Christmas Day', 'date': '2024-12-25'},
    {'name': 'Boxing Day', 'date': '2024-12-26'},
    # 2025
    {'name': 'New Year', 'date': '2025-01-01'},
    {'name': 'Good Friday', 'date': '2025-04-18'},
    {'name': 'Easter Monday', 'date': '2025-04-21'},
    {'name': 'Labour Day', 'date': '2025-05-01'},
    {'name': 'Ascension Day', 'date': '2025-05-29'},
    {'name': 'Whit Monday', 'date': '2025-06-09'},
    {'name': 'German Unity Day', 'date': '2025-10-03'},
    {'name': 'Christmas Day', 'date': '2025-12-25'},
    {'name': 'Boxing Day', 'date': '2025-12-26'},
    # 2026
    {'name': 'New Year', 'date': '2026-01-01'},
    {'name': 'Good Friday', 'date': '2026-04-03'},
    {'name': 'Easter Monday', 'date': '2026-04-06'},
    {'name': 'Labour Day', 'date': '2026-05-01'},
]


def fetch_bundesliga_matches():
    """
    Fetch Bundesliga match data from OpenLigaDB (free API).

    Returns:
        DataFrame with match data
    """
    matches = []
    base_url = "https://api.openligadb.de/getmatchdata/bl1"

    # Fetch multiple seasons
    for season in [2023, 2024, 2025]:
        print(f"  Fetching Bundesliga {season}/{season+1}...")
        try:
            url = f"{base_url}/{season}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            for match in data:
                # Check if Berlin teams involved
                team1 = match.get('team1', {}).get('teamName', '')
                team2 = match.get('team2', {}).get('teamName', '')

                berlin_home = None
                if 'Hertha' in team1:
                    berlin_home = 'Hertha BSC'
                elif 'Union' in team1:
                    berlin_home = 'Union Berlin'

                if berlin_home:
                    match_date = match.get('matchDateTime', '')
                    if match_date:
                        matches.append({
                            'date': match_date[:10],
                            'time': match_date[11:16] if len(match_date) > 11 else None,
                            'event_type': 'football_home',
                            'event_name': f"{team1} vs {team2}",
                            'team': berlin_home,
                            'stadium': BERLIN_TEAMS[berlin_home]['stadium'],
                            'affected_neighborhoods': ', '.join(BERLIN_TEAMS[berlin_home]['neighborhoods']),
                            'expected_attendance': BERLIN_TEAMS[berlin_home]['capacity'] * 0.8,
                        })
        except Exception as e:
            print(f"    Error fetching {season}: {e}")

    return pd.DataFrame(matches)


def create_holiday_dataframe():
    """Create DataFrame of school holidays."""
    records = []
    for holiday in SCHOOL_HOLIDAYS:
        start = pd.to_datetime(holiday['start'])
        end = pd.to_datetime(holiday['end'])

        # Create entry for end of holiday (return date)
        records.append({
            'date': (end + timedelta(days=1)).strftime('%Y-%m-%d'),
            'event_type': 'holiday_end',
            'event_name': f"{holiday['name']} ends",
            'holiday_duration': (end - start).days + 1,
            'affected_neighborhoods': 'All',
            'expected_impact': 'vacation_return_effect',
        })

        # Create entries for during holiday
        records.append({
            'date': start.strftime('%Y-%m-%d'),
            'event_type': 'holiday_start',
            'event_name': f"{holiday['name']} starts",
            'holiday_duration': (end - start).days + 1,
            'affected_neighborhoods': 'All',
            'expected_impact': 'reduced_demand',
        })

    return pd.DataFrame(records)


def create_events_dataframe():
    """Create DataFrame of major events."""
    records = []
    for event in MAJOR_EVENTS:
        start = pd.to_datetime(event['date'])
        for day in range(event['duration']):
            records.append({
                'date': (start + timedelta(days=day)).strftime('%Y-%m-%d'),
                'event_type': event['type'],
                'event_name': event['name'],
                'day_of_event': day + 1,
                'total_days': event['duration'],
                'affected_neighborhoods': 'Central Berlin',
            })

    return pd.DataFrame(records)


def create_holidays_dataframe():
    """Create DataFrame of public holidays."""
    records = []
    for holiday in PUBLIC_HOLIDAYS:
        records.append({
            'date': holiday['date'],
            'event_type': 'public_holiday',
            'event_name': holiday['name'],
            'affected_neighborhoods': 'All',
            'stores_closed': True,
        })
    return pd.DataFrame(records)


def save_data(df, filename_prefix='events_berlin'):
    """Save data to CSV with timestamp."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"{filename_prefix}_{timestamp}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(filepath, index=False)
    print(f"\nSaved to: {filepath}")
    return filepath


def main():
    """Main collection routine."""
    print("=" * 60)
    print("AGRICOM - Events Data Collection")
    print("=" * 60)

    all_events = []

    # Bundesliga matches
    print("\n1. Fetching Bundesliga matches...")
    try:
        df_matches = fetch_bundesliga_matches()
        all_events.append(df_matches)
        print(f"   Found {len(df_matches)} home matches")
    except Exception as e:
        print(f"   Error: {e}")

    # School holidays
    print("\n2. Processing school holidays...")
    df_holidays = create_holiday_dataframe()
    all_events.append(df_holidays)
    print(f"   Found {len(df_holidays)} holiday events")

    # Major events
    print("\n3. Processing major events...")
    df_events = create_events_dataframe()
    all_events.append(df_events)
    print(f"   Found {len(df_events)} event days")

    # Public holidays
    print("\n4. Processing public holidays...")
    df_public = create_holidays_dataframe()
    all_events.append(df_public)
    print(f"   Found {len(df_public)} public holidays")

    # Combine all
    print("\n5. Combining all events...")
    combined = pd.concat(all_events, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    combined = combined.sort_values('date')

    # Save
    filepath = save_data(combined)

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total events: {len(combined)}")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print("\nEvent types:")
    print(combined['event_type'].value_counts())


if __name__ == "__main__":
    main()
