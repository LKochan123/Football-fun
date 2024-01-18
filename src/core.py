import pandas as pd
import numpy as np

# Complete with your need
TEAMS_IN_LEAGUE = 20
NEWJOINER_AVG_POINTS = 1.05

cols = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY"]

# Creating target column: 0 - Home_Team loss, 1 - Home_Team win, 2 - draw
def data_cleaner(df: pd.DataFrame):
    df_copy = df.copy()
    df_copy["target"] = np.where(df_copy["FTR"] == "H", 0, np.where(df_copy["FTR"] == "A", 2, 1))
    return df_copy

def get_all_team_matches(df: pd.DataFrame, name: str):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    HT_matches = df.groupby("HomeTeam")
    AT_matches = df.groupby("AwayTeam")

    all_team_matches = pd.concat([HT_matches.get_group(name), AT_matches.get_group(name)])
    all_team_matches.sort_values(by="Date", inplace=True)
    return all_team_matches

def calc_curr_team_goal_difference(df: pd.DataFrame, window=5):
    df_copy = df.copy()
    HT_goals_scored = df_copy.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    HT_goals_conceded = df_copy.groupby('HomeTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    df_copy[f"H_gd_{window}"] = (HT_goals_scored - HT_goals_conceded)

    AT_goals_scored = df_copy.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    AT_goals_conceded = df_copy.groupby('AwayTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    df_copy[f"A_gd_{window}"] = (AT_goals_scored - AT_goals_conceded)

    return df_copy

def calc_effectivnes_team(df: pd.DataFrame, window=5):
    df_copy = df.copy()
    HT_goals_scored = df_copy.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    HT_shots_target = df_copy.groupby('HomeTeam')['HST'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    df_copy[f"H_eff_{window}"] = np.where(HT_shots_target != 0, np.round((HT_goals_scored / HT_shots_target), 2), 0)

    AT_goals_scored = df_copy.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    AT_shots_target = df_copy.groupby('AwayTeam')['AST'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    df_copy[f"A_eff_{window}"] = np.where(AT_shots_target != 0, np.round((AT_goals_scored / AT_shots_target), 2), 0)

    return df_copy

def calc_curr_mean_statistic(df: pd.DataFrame, cols: list[str], window=5):
    df_copy = df.copy()
    for col in cols:
        rolling_col_name = col + '_avg' + f'_{window}'
        team_col = 'HomeTeam' if col[0] == 'H' else 'AwayTeam'

        rolling_value = df_copy.groupby(team_col)[col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        df_copy[rolling_col_name] = np.round(rolling_value, 2).fillna(0)

    return df_copy

def calc_avg_points(df: pd.DataFrame, window=5):
    df_copy = df.copy()

    df_copy['HT_Points'] = df_copy['FTR'].apply(lambda x: 3 if x == 'H' else 1 if x == 'D' else 0)
    df_copy['AT_Points'] = df_copy['FTR'].apply(lambda x: 3 if x == 'A' else 1 if x == 'D' else 0)

    df_copy['Cumulative_Points_Home'] = df_copy.groupby('HomeTeam')['HT_Points'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    df_copy['Cumulative_Matches_Home'] = df_copy.groupby('HomeTeam')['HT_Points'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).count())
    df_copy[f'HPTS_avg_{window}'] = np.round((df_copy['Cumulative_Points_Home'] / df_copy['Cumulative_Matches_Home']), 2)

    df_copy['Cumulative_Points_Away'] = df_copy.groupby('AwayTeam')['AT_Points'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    df_copy['Cumulative_Matches_Away'] = df_copy.groupby('AwayTeam')['AT_Points'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).count())
    df_copy[f'APTS_avg_{window}'] = np.round((df_copy['Cumulative_Points_Away'] / df_copy['Cumulative_Matches_Away']), 2)

    df_copy = df_copy.drop(['HT_Points', 'AT_Points', 'Cumulative_Points_Home', 'Cumulative_Matches_Home', 'Cumulative_Points_Away', 'Cumulative_Matches_Away'], 
                           axis=1, errors='ignore')

    return df_copy

# All neccesary functions to transform data combined
def data_transformation(df: pd.DataFrame, cols: list[str], window=5):
    df_copy = df.copy()
    df_copy = calc_avg_points(df_copy, window)
    df_copy = calc_curr_team_goal_difference(df_copy, window)
    df_copy = calc_effectivnes_team(df_copy, window)
    df_copy = calc_curr_mean_statistic(df_copy, cols, window)
    df_copy = data_cleaner(df_copy)
    return df_copy

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

def calc_team_status(current_season_df: pd.DataFrame, last_seaons_df: list[pd.DataFrame], team: str):
    N = (TEAMS_IN_LEAGUE - 1) * 2

    if is_team_newcomer(current_season_df, last_seaons_df[0], team):
        return NEWJOINER_AVG_POINTS
    else:
        points_counter, seasons_in_league = 0, 0
        for previous_season in last_seaons_df:
            final_table = calculate_final_table(previous_season)
            if team in final_table['Team'].values:
                points_counter += final_table.loc[final_table['Team'] == team, 'Points'].values[0]
                seasons_in_league +=1 

        return np.round(points_counter / (N * seasons_in_league), 2)
    
def create_teams_status_dict(current_season_df: pd.DataFrame, last_seasons_df: list[pd.DataFrame]):
    teams = set(current_season_df['HomeTeam'])
    return { team: calc_team_status(current_season_df, last_seasons_df, team) for team in teams }

def calc_h2h_stats(current_season_df: pd.DataFrame, last_seasons_df: list[pd.DataFrame]):
    curr_df = current_season_df.copy()
    curr_df['H_H2H'], curr_df['A_H2H'] = 0, 0

    for index, row in curr_df.iterrows():
        team_H, team_A = row['HomeTeam'], row['AwayTeam']

        matches = pd.concat([df[((df['HomeTeam'] == team_A) & (df['AwayTeam'] == team_H)) | 
                                ((df['HomeTeam'] == team_H) & (df['AwayTeam'] == team_A))] for df in last_seasons_df])
        
        if not matches.empty:
            points_A = (3 * len(matches[(matches['HomeTeam'] == team_A) & (matches['FTR'] == 'H')]) + 
                        3 * len(matches[(matches['AwayTeam'] == team_A) & (matches['FTR'] == 'A')]) + 
                        len(matches[((matches['HomeTeam'] == team_A) | (matches['AwayTeam'] == team_A)) & (matches['FTR'] == 'D')]))

            points_H = (3 * len(matches[(matches['HomeTeam'] == team_H) & (matches['FTR'] == 'H')]) + 
                        3 * len(matches[(matches['AwayTeam'] == team_H) & (matches['FTR'] == 'A')]) + 
                        len(matches[((matches['HomeTeam'] == team_H) | (matches['AwayTeam'] == team_H)) & (matches['FTR'] == 'D')]))

            curr_df.at[index, 'H_H2H'] = np.round(points_H / matches.shape[0], 2)
            curr_df.at[index, 'A_H2H'] = np.round(points_A / matches.shape[0], 2)   
        else:
            curr_df.at[index, 'H_H2H'] = NEWJOINER_AVG_POINTS if is_team_newcomer(current_season_df, last_seasons_df[0], team_H) else calc_team_status(current_season_df, last_seasons_df, team_H)
            curr_df.at[index, 'A_H2H'] = NEWJOINER_AVG_POINTS if is_team_newcomer(current_season_df, last_seasons_df[0], team_A) else calc_team_status(current_season_df, last_seasons_df, team_A)

    return curr_df

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
    newjoiners_all = []
    for i in range(1, len(seasons_file_name)):
        last_season = pd.read_csv(f"../data/raw/{league_name}/{seasons_file_name[i-1]}.csv")
        curr_season = pd.read_csv(f"../data/raw/{league_name}/{seasons_file_name[i]}.csv")
        curr_newjoiners_stats = find_newcomer_teams_statistics(last_season, curr_season)
        newjoiners_all.append(curr_newjoiners_stats)
    return newjoiners_all

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