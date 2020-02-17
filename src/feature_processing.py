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


