"""
For this competition, this module will be responsible for generating training data
before using feature generator
"""
import pandas as pd
from . import dispatcher, utils


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


if __name__ == '__main__':
    tourney_result = pd.read_csv(f'{dispatcher.WDSTAGE1_WNCAA_COMPACT_RESULTS}')
    tourney_seed = pd.read_csv(f'{dispatcher.WDSTAGE1_WNCAA_SEEDS}')
    tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
    tourney_result = baseline_merge_tourney_results_seed(tourney_result, tourney_seed)
    tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))
    tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))

    season_result = pd.read_csv(f'{dispatcher.WDSTAGE1_WREGULAR_COMPACT_RESULTS}')
    season_result = baseline_process_season_results(season_result)

    season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()
    tourney_result = baseline_merge_tourney_results_season_results(tourney_result, season_score)

    tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)
    tourney_win_result.rename(columns={'WSeed': 'Seed1', 'LSeed': 'Seed2', 'WScoreT': 'ScoreT1', 'LScoreT': 'ScoreT2'},
                              inplace=True)

    tourney_lose_result = tourney_win_result.copy()
    tourney_lose_result['Seed1'] = tourney_win_result['Seed2']
    tourney_lose_result['Seed2'] = tourney_win_result['Seed1']
    tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']
    tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']

    tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
    tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']
    tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
    tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']

    tourney_win_result['result'] = 1
    tourney_lose_result['result'] = 0
    tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)

    utils.check_make_dirs(dispatcher.CUSTOM_DATA_FOLDER)
    tourney_result.to_csv(dispatcher.CUSTOM_BASELINE_FEATURE_CSV, index=False)





