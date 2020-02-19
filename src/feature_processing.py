"""
For this competition, this module will be responsible for generating training data
before using feature generator
"""
import pandas as pd


# Baseline feature processing
# Thanks to this amazing kernel: https://www.kaggle.com/hiromoon166/2020-women-s-starter-kernel

def baseline_merge_tourney_results_seed(tourney_result, tourney_seed):
    tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'],
                              right_on=['Season', 'TeamID'], how='left')
    tourney_result.rename(columns={'Seed': 'WSeed'}, inplace=True)
    tourney_result = tourney_result.drop('TeamID', axis=1)
    tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'],
                              right_on=['Season', 'TeamID'], how='left')
    tourney_result.rename(columns={'Seed': 'LSeed'}, inplace=True)
    tourney_result = tourney_result.drop('TeamID', axis=1)

    return tourney_result


def get_seed(x):
    return int(x[1:3])


def baseline_process_season_results(season_result):
    season_win_result = season_result[['Season', 'WTeamID', 'WScore']]
    season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]
    season_win_result.rename(columns={'WTeamID': 'TeamID', 'WScore': 'Score'}, inplace=True)
    season_lose_result.rename(columns={'LTeamID': 'TeamID', 'LScore': 'Score'}, inplace=True)
    season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)

    return season_result


def baseline_merge_tourney_results_season_results(tourney_result, season_score):
    tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'],
                              right_on=['Season', 'TeamID'], how='left')
    tourney_result.rename(columns={'Score': 'WScoreT'}, inplace=True)
    tourney_result = tourney_result.drop('TeamID', axis=1)
    tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'],
                              right_on=['Season', 'TeamID'], how='left')
    tourney_result.rename(columns={'Score': 'LScoreT'}, inplace=True)
    tourney_result = tourney_result.drop('TeamID', axis=1)

    return tourney_result




