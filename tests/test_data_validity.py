import pandas as pd
import unittest
import os

# Complete data with your needs
# Premier_league - E0, La_liga - SP1, Serie_A - I1, Ligue_1 - F1
TEAMS_IN_LEAGUE = 20
LEAGUE_NAME = "Premier_league"
LEAGUE_CODE = "E0"
FIRST_SEASON_YEAR, LAST_SEASON_YEAR = 5, 23
MATCHES_IN_SEASON = TEAMS_IN_LEAGUE * (TEAMS_IN_LEAGUE - 1)

selected_columns = [
'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HR', 'AR'
]

def check_selected_columns(file_name):
    season = get_season(file_name)

    for column in selected_columns:
        assert column in season.columns, f"Column '{column}' not found in {file_name}"

def check_season_shape(file_name):
    N = (MATCHES_IN_SEASON, len(selected_columns))
    season = get_season(file_name)
    assert season[selected_columns].shape == N

def check_number_of_games_for_each_team(file_name):
    N = TEAMS_IN_LEAGUE - 1
    season = get_season(file_name)

    for team, games in season["AwayTeam"].value_counts().items():
        assert games == N, f"Team '{team}' does not have {TEAMS_IN_LEAGUE - 1} Away games in {file_name}"

def check_isnull_elements(file_name):
    season = get_season(file_name)
    assert not season[selected_columns].isnull().values.any(), f"Some values are missing in file {file_name}"

def check_result_of_game(file_name):
    season = get_season(file_name)
    assert (season.apply(check_results_consistency, axis=1)).all(), f"Missmatch with result of the game in {file_name}"

def check_nonnegativity_values(file_name):
    cols_to_check = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HR', 'AR']
    season = get_season(file_name)

    assert (season[cols_to_check] >= 0).all(axis=None), f"Some values are negative in {file_name} in column where it shouldn't."
    assert season[cols_to_check].applymap(lambda x: isinstance(x, int)).all().all(), f"Some values are not integer in file {file_name} in column where it shouldn't."

def check_results_consistency(row):
    if row['FTHG'] > row['FTAG'] and row['FTR'] == 'H':
        return True
    elif row['FTHG'] < row['FTAG'] and row['FTR'] == 'A':
        return True
    elif row['FTHG'] == row['FTAG'] and row['FTR'] == 'D':
        return True
    else:
        return False
    
def get_season(file_name):
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', LEAGUE_NAME, f'{file_name}.csv'))
    return pd.read_csv(path)
    
class TestFootballData(unittest.TestCase):

    def test_selected_columns(self):
        self.run_test(check_selected_columns, "selected columns")

    def test_shape(self):
        self.run_test(check_season_shape, "shape of data")

    def test_number_of_games(self):
        self.run_test(check_number_of_games_for_each_team, "number of games")

    def test_missing_values(self):
        self.run_test(check_isnull_elements, "null values")

    def test_game_results(self):
        self.run_test(check_result_of_game, "games result")

    def test_nonnegativity_values(self):
        self.run_test(check_nonnegativity_values, "non negative values")

    def run_test(self, test_function, test_name):
        successful_tests, total_tests = 0, 0
        files_name = [f"{LEAGUE_CODE}_{str(year).zfill(2)}_{str(year+1).zfill(2)}" for year in range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR)]

        for file in files_name:
            try:
                test_function(file)
                successful_tests += 1
            except AssertionError as e:
                print(f"{test_name} for {file} failed: {e}")
        
            total_tests += 1
        
        print(f"Summary: {successful_tests}/{total_tests} files passed {test_name} test.")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)