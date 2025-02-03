
from config import LOADING_TIME_BEGINNING, TIME_BEGINNING ,TIME_END , seasons, years_padded  , SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, DataYearly, SamplesCoordinates_Seasonally, MatrixCoordinates_1mil_Seasonally, DataSeasonally ,file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC 
import pandas as pd
import numpy as np

def get_time_range(TIME_BEGINNING= TIME_BEGINNING, TIME_END=TIME_END, seasons=seasons, years_padded=years_padded):
    # Define seasons list for matching
    seasons_list = ['winter', 'spring', 'summer', 'autumn']

    # Check if TIME_BEGINNING is a season
    is_season = any(season in TIME_BEGINNING.lower() for season in seasons_list)

    if is_season:
        # Handle seasons case
        start_idx = next(i for i, season in enumerate(seasons) 
                        if TIME_BEGINNING.lower() in season.lower())
        end_idx = next(i for i, season in enumerate(seasons) 
                      if TIME_END.lower() in season.lower())

        # Get the subset including both start and end
        return seasons[start_idx:end_idx + 1]
    else:
        # Handle years case
        start_idx = years_padded.index(TIME_BEGINNING)
        end_idx = years_padded.index(TIME_END)

        # Get the subset including both start and end
        return years_padded[start_idx:end_idx + 1]


def process_paths_yearly(path, year, seen_years):
    if 'Elevation' in path:
        return path
    elif 'MODIS_NPP' in path:
        paths = []
        # Add current year
        if year not in seen_years:
            seen_years.add(year)
            paths.append(f"{path}/{year}")
        # Add previous year
        prev_year = str(int(year) - 1)
        if prev_year not in seen_years:
            seen_years.add(prev_year)
            paths.append(f"{path}/{prev_year}")
        return paths
    else:
        return f"{path}/{year}"

def create_path_arrays_yearly(SamplesCoordinates_Yearly, DataYearly, selected_years):
    seen_years_samples = set()
    seen_years_data = set()

    samples_coordinates_array_path = [
        processed_path
        for idx, base_path in enumerate(SamplesCoordinates_Yearly)
        for year in selected_years
        if idx < len(SamplesCoordinates_Yearly)
        if (processed_path := process_paths_yearly(base_path, year, seen_years_samples)) is not None
    ]

    data_yearly_array_path = [
        processed_path
        for idx, base_path in enumerate(DataYearly)
        for year in selected_years
        if idx < len(DataYearly)
        if (processed_path := process_paths_yearly(base_path, year, seen_years_data)) is not None
    ]

    return samples_coordinates_array_path, data_yearly_array_path


def process_paths(path, season, seen_years):
    if 'Elevation' in path:
        return path
    elif 'MODIS_NPP' in path:
        year = season.split('_')[0][:4]  # Get year from season
        paths = []
        # Add current year
        if year not in seen_years:
            seen_years.add(year)
            paths.append(f"{path}/{year}")
        # Add previous year
        prev_year = str(int(year) - 1)
        if prev_year not in seen_years:
            seen_years.add(prev_year)
            paths.append(f"{path}/{prev_year}")
        return paths
    else:
        return f"{path}/{season}"

def create_path_arrays(SamplesCoordinates_Seasonally, DataSeasonally, selected_seasons):
    seen_years_samples = set()
    seen_years_data = set()

    samples_coordinates_array_path = [
        processed_path
        for idx, base_path in enumerate(SamplesCoordinates_Seasonally)
        for season in selected_seasons
        if idx < len(SamplesCoordinates_Seasonally)
        if (processed_path := process_paths(base_path, season, seen_years_samples)) is not None
    ]

    data_seasons_array_path = [
        processed_path
        for idx, base_path in enumerate(DataSeasonally)
        for season in selected_seasons
        if idx < len(DataSeasonally)
        if (processed_path := process_paths(base_path, season, seen_years_data)) is not None
    ]
    
    return samples_coordinates_array_path, data_seasons_array_path



def separate_and_add_data(LOADING_TIME_BEGINNING=LOADING_TIME_BEGINNING, TIME_END=TIME_END, seasons=seasons, years_padded=years_padded, 
                         SamplesCoordinates_Yearly=SamplesCoordinates_Yearly, DataYearly=DataYearly,
                         SamplesCoordinates_Seasonally=SamplesCoordinates_Seasonally, DataSeasonally=DataSeasonally):

    # Define seasons list for matching
    seasons_list = ['winter', 'spring', 'summer', 'autumn']

    # Check if LOADING_TIME_BEGINNING is a season
    is_season = any(season in LOADING_TIME_BEGINNING.lower() for season in seasons_list)

    if is_season:
        # Handle seasons case
        start_idx = next(i for i, season in enumerate(seasons) 
                        if LOADING_TIME_BEGINNING.lower() in season.lower())
        end_idx = next(i for i, season in enumerate(seasons) 
                      if TIME_END.lower() in season.lower())

        # Get the seasonal range
        selected_seasons = seasons[start_idx:end_idx + 1]


        # Add seasonal data pairs
        return create_path_arrays(SamplesCoordinates_Seasonally, DataSeasonally, selected_seasons)
    else:
        start_idx = years_padded.index(LOADING_TIME_BEGINNING)
        end_idx = years_padded.index(TIME_END)
        selected_years = years_padded[start_idx:end_idx + 1]
        return create_path_arrays_yearly(SamplesCoordinates_Yearly, DataYearly, selected_years)

def add_season_column(dataframe):
    seasons_months = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11]
    }

    month_to_season = {
        month: season
        for season, months in seasons_months.items()
        for month in months
    }

    dataframe['survey_date'] = pd.to_datetime(dataframe['survey_date'])

    def get_season_year(row):
        if pd.isna(row['survey_date']):
            return None

        month = row['survey_date'].month
        year = row['survey_date'].year

        if month == 12:
            year += 1

        season = month_to_season.get(month)
        if season:
            return f"{year}_{season}"
        return None

    valid_dates_mask = dataframe['survey_date'] >= '2000-01-01'
    dataframe['season'] = None
    dataframe.loc[valid_dates_mask, 'season'] = (
        dataframe[valid_dates_mask].apply(get_season_year, axis=1)
    )

    return dataframe

def filter_dataframe(time_beginning, time_end, max_oc=150):
    # Read and prepare data
    df = pd.read_excel(file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC)
    df = add_season_column(df)

    # Convert columns to numeric
    df['GPS_LONG'] = pd.to_numeric(df['GPS_LONG'], errors='coerce')
    df['GPS_LAT'] = pd.to_numeric(df['GPS_LAT'], errors='coerce')
    df['OC'] = pd.to_numeric(df['OC'], errors='coerce')

    # Basic data quality mask
    quality_mask = (
        (df['OC'] <= max_oc) &
        df['GPS_LONG'].notna() &
        df['GPS_LAT'].notna() &
        df['OC'].notna()
    )

    # Check if time_beginning contains a season
    seasons = ['winter', 'spring', 'summer', 'autumn']
    is_season = any(season in time_beginning.lower() for season in seasons)

    if is_season:
        # Create a list of all valid seasons between time_beginning and time_end
        start_year, start_season = time_beginning.split('_')
        end_year, end_season = time_end.split('_')
        start_year = int(start_year)
        end_year = int(end_year)

        valid_seasons = []
        current_year = start_year
        season_order = ['winter', 'spring', 'summer', 'autumn']
        start_idx = season_order.index(start_season)
        end_idx = season_order.index(end_season)

        while current_year <= end_year:
            if current_year == start_year:
                season_start = start_idx
            else:
                season_start = 0

            if current_year == end_year:
                season_end = end_idx
            else:
                season_end = len(season_order) - 1

            for season in season_order[season_start:season_end + 1]:
                valid_seasons.append(f"{current_year}_{season}")

            current_year += 1

        # Filter using the valid seasons list
        filtered_df = df[
            df['season'].isin(valid_seasons) &
            quality_mask
        ]
    else:
        # Filter by year range
        start_year = int(time_beginning)
        end_year = int(time_end)
        filtered_df = df[
            (df['year'].between(start_year, end_year, inclusive='both')) &
            quality_mask
        ]

    print(f"Initial shape: {df.shape}")
    print(f"Final filtered shape: {filtered_df.shape}")

    if filtered_df.empty:
        print("\nDebug information:")
        print("NaN counts:", df[['GPS_LONG', 'GPS_LAT', 'OC', 'survey_date']].isna().sum())
        print(f"OC range: {df['OC'].min()} to {df['OC'].max()}")

    return filtered_df
