import pandas as pd
import numpy as np
import stats as s

# Complete with your need
TEAMS_IN_LEAGUE = 20
NEWJOINER_AVG_POINTS = 1.05

cols = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY"]

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
    df_copy = s.data_cleaner(df_copy)
    return df_copy

def calc_team_status(current_season_df: pd.DataFrame, last_seaons_df: list[pd.DataFrame], team: str):
    N = (TEAMS_IN_LEAGUE - 1) * 2

    if s.is_team_newcomer(current_season_df, last_seaons_df[0], team):
        return NEWJOINER_AVG_POINTS
    else:
        points_counter, seasons_in_league = 0, 0
        for previous_season in last_seaons_df:
            final_table = s.calculate_final_table(previous_season)
            if team in final_table['Team'].values:
                points_counter += final_table.loc[final_table['Team'] == team, 'Points'].values[0]
                seasons_in_league += 1 

        return np.round(points_counter / (N * seasons_in_league), 2) if seasons_in_league > 0 else 0
    
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
            curr_df.at[index, 'H_H2H'] = NEWJOINER_AVG_POINTS if s.is_team_newcomer(current_season_df, last_seasons_df[0], team_H) else calc_team_status(current_season_df, last_seasons_df, team_H)
            curr_df.at[index, 'A_H2H'] = NEWJOINER_AVG_POINTS if s.is_team_newcomer(current_season_df, last_seasons_df[0], team_A) else calc_team_status(current_season_df, last_seasons_df, team_A)

    return curr_df