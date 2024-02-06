import pandas as pd
import numpy as np

#Complete with your needs
TEAMS_IN_LEAGUE = 20

# Creating target column: 0 - Home_Team loss, 1 - Home_Team win, 2 - draw
def data_cleaner(df: pd.DataFrame):
    df_copy = df.copy()
    df_copy["target"] = np.where(df_copy["FTR"] == "H", 0, np.where(df_copy["FTR"] == "A", 2, 1))
    return df_copy

def find_season_name(season_start_year: int, league_code: str):
    if season_start_year < 9:
        return f"{league_code}_0{season_start_year}_0{season_start_year + 1}"
    elif season_start_year == 9:
        return f"{league_code}_0{season_start_year}_{season_start_year + 1}"
    else:
        return f"{league_code}_{season_start_year}_{season_start_year+ 1}"

def get_all_team_matches(df: pd.DataFrame, name: str):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    HT_matches = df.groupby("HomeTeam")
    AT_matches = df.groupby("AwayTeam")

    all_team_matches = pd.concat([HT_matches.get_group(name), AT_matches.get_group(name)])
    all_team_matches.sort_values(by="Date", inplace=True)
    return all_team_matches

def calculate_final_table(df: pd.DataFrame):
    team_points = {}

    for _, row in df.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        ftr = row['FTR']

        if home_team not in team_points:
            team_points[home_team] = 0
        if away_team not in team_points:
            team_points[away_team] = 0

        if ftr == 'H':
            team_points[home_team] += 3
        elif ftr == 'A':
            team_points[away_team] += 3
        elif ftr == 'D':
            team_points[home_team] += 1
            team_points[away_team] += 1

    points_df = pd.DataFrame(list(team_points.items()), columns=['Team', 'Points'])
    sorted_points_df = points_df.sort_values(by='Points', ascending=False).reset_index(drop=True)

    return sorted_points_df

def get_team_total_points(df: pd.DataFrame, team: str):
    final_table = calculate_final_table(df)
    team_row = final_table.loc[final_table['Team'] == team, 'Points']
    return team_row.values[0] if not team_row.empty else -1
        
def get_team_avg_points(df: pd.DataFrame, team: str):
    N = (TEAMS_IN_LEAGUE - 1) * 2 
    final_table = calculate_final_table(df)
    team_row = final_table.loc[final_table['Team'] == team, 'Points']
    return np.round(team_row.values[0] / N, 2) if not team_row.empty else -1
        
def get_team_position(df: pd.DataFrame, team: str):
    final_table = calculate_final_table(df)
    team_row = final_table.loc[final_table['Team'] == team]
    return int(team_row.index[0] + 1 if not team_row.empty else -1)

def is_team_newcomer(last_season_df: pd.DataFrame, current_season_df: pd.DataFrame, team: str):
    last_season_table, current_season_table = calculate_final_table(last_season_df), calculate_final_table(current_season_df)
    return team in current_season_table['Team'].values and team not in last_season_table['Team'].values

def find_newcomer_teams_statistics(last_season_df: pd.DataFrame, current_season_df: pd.DataFrame):
    last_season_table, current_season_table = calculate_final_table(last_season_df), calculate_final_table(current_season_df)
    newcomer_teams = [team for team in list(current_season_table['Team']) if team not in list(last_season_table['Team'])]
    return { 
        name : { 
            'Points': get_team_total_points(current_season_df, name), 
            'Avg': get_team_avg_points(current_season_df, name),
            'Position': get_team_position(current_season_df, name)
        } for name in newcomer_teams
    }

def find_all_newjoiners(seasons_file_name: list[str], league_name: str):
    newjoiners_all, processed_data = [], []
    
    for i in range(1, len(seasons_file_name)):
        last_season = pd.read_csv(f"../data/raw/{league_name}/{seasons_file_name[i-1]}")
        curr_season = pd.read_csv(f"../data/raw/{league_name}/{seasons_file_name[i]}")
        curr_newjoiners_stats = find_newcomer_teams_statistics(last_season, curr_season)
        newjoiners_all.append(curr_newjoiners_stats)
    
    for season, newcomers in enumerate(newjoiners_all, start=2005):
        short_season = f"{season % 100:02d}/{(season % 100) + 1:02d}"
        for team, stats in newcomers.items():
            processed_data.append({'Season': short_season, 'Team': team, **stats})

    return pd.DataFrame(processed_data)

def calc_HT_cards(df: pd.DataFrame, card = 'Y', N = 9):
    cards_results = {i: { 'wins': 0, 'losses': 0, 'draws': 0 } for i in range(1, N)}

    if card not in ['Y', 'R']:
        raise ValueError("Argument 'card' must be 'Y' or 'R'.")

    for _, row in df.iterrows():
        for num in range(1, N):
            if num == row[f'H{card}']:
                if row['FTHG'] > row['FTAG']:
                    cards_results[num]['wins'] += 1
                elif row['FTHG'] == row['FTAG']:
                    cards_results[num]['draws'] += 1
                else:
                    cards_results[num]['losses'] += 1

    return cards_results


def calc_AT_cards(df: pd.DataFrame, card = 'Y', N = 9):
    cards_results = {i: { 'wins': 0, 'losses': 0, 'draws': 0 } for i in range(1, N)}

    if card not in ['Y', 'R']:
        raise ValueError("Argument 'card' must be 'Y' or 'R'.")

    for _, row in df.iterrows():
        for num in range(1, N):
            if num == row[f'A{card}']:
                if row['FTAG'] > row['FTHG']:
                    cards_results[num]['wins'] += 1
                elif row['FTHG'] == row['FTAG']:
                    cards_results[num]['draws'] += 1
                else:
                    cards_results[num]['losses'] += 1

    return cards_results


def count_results(df: pd.DataFrame):
    home_wins, away_wins, draws = 0, 0, 0
    
    for _, row in df.iterrows():
        result = row['FTR']
        if result == 'H':
            home_wins += 1
        elif result == 'A':
            away_wins += 1
        elif result == 'D':
            draws += 1
    
    return home_wins, away_wins, draws

def count_seasons_per_team(df: pd.DataFrame):
    team_seasons = {}
    
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        date = pd.to_datetime(row['Date'], format='%d/%m/%y')
        
        season_start = date.month
        season_end = 5 if date.month <= 5 else 6
        
        if season_start <= date.month <= season_end:
            season = f"{date.year - 1}/{date.year}"
        else:
            season = f"{date.year}/{date.year + 1}"
        
        if home_team in team_seasons:
            team_seasons[home_team].add(season)
        else:
            team_seasons[home_team] = {season}
        
        if away_team in team_seasons:
            team_seasons[away_team].add(season)
        else:
            team_seasons[away_team] = {season}
    
    result = {team: len(seasons) for team, seasons in team_seasons.items()}
    return result