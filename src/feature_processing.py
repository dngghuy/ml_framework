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
    # Ver1
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

    tourney_win_result = tourney_result.drop([ 'WTeamID', 'LTeamID'], axis=1)
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

    tourney_win_result['target'] = 1
    tourney_lose_result['target'] = 0
    tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)

    utils.check_make_dirs(dispatcher.CUSTOM_DATA_FOLDER)
    tourney_result.to_csv(dispatcher.CUSTOM_BASELINE_FEATURE_CSV, index=False)
    print('Prepare baseline test df')
    test_df = pd.read_csv(dispatcher.WSAMPLE_SUBMISSION)
    test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
    test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
    test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))
    test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    test_df.rename(columns={'Seed': 'Seed1'}, inplace=True)
    test_df = test_df.drop('TeamID', axis=1)
    test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
    test_df.rename(columns={'Seed': 'Seed2'}, inplace=True)
    test_df = test_df.drop('TeamID', axis=1)
    test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    test_df.rename(columns={'Score': 'ScoreT1'}, inplace=True)
    test_df = test_df.drop('TeamID', axis=1)
    test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
    test_df.rename(columns={'Score': 'ScoreT2'}, inplace=True)
    test_df = test_df.drop('TeamID', axis=1)
    test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))
    test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))
    test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
    test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']
    test_df = test_df.drop(['ID', 'Pred', 'WTeamID', 'LTeamID'], axis=1)
    test_df.to_csv(dispatcher.CUSTOM_BASELINE_TEST_CSV, index=False)