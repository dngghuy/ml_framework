import pathlib
from sklearn import ensemble

# ### CONSTANT SETTINGS ###
# DATA PATH SETTINGS
DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'
VISUALIZE_FOLDER = pathlib.Path(__file__).parent.parent / 'visualizations'
MODELS_FOLDER = pathlib.Path(__file__).parent.parent / 'models'


# PER-PRJ SETTINGS
# Kaggle basketball women
KAGGLE_GBW_FOLDER = DATA_PATH / 'kaggle_GBW'
WDATA_STAGE1 = KAGGLE_GBW_FOLDER / 'WDataFiles_Stage1'
WEVENTS_2015 = KAGGLE_GBW_FOLDER / 'WEvents2015.csv'
WEVENTS_2016 = KAGGLE_GBW_FOLDER / 'WEvents2016.csv'
WEVENTS_2017 = KAGGLE_GBW_FOLDER / 'WEvents2017.csv'
WEVENTS_2018 = KAGGLE_GBW_FOLDER / 'WEvents2018.csv'
WEVENTS_2019 = KAGGLE_GBW_FOLDER / 'WEvents2019.csv'
WPLAYERS = KAGGLE_GBW_FOLDER / 'WPlayers.csv'
WSAMPLE_SUBMISSION = KAGGLE_GBW_FOLDER / 'WSampleSubmissionStage1_2020.csv'

WDSTAGE1_CITIES = WDATA_STAGE1 / 'Cities.csv'
WDSTAGE1_CONFERENCES = WDATA_STAGE1 / 'Conferences.csv'
WDSTAGE1_WGameCities = WDATA_STAGE1 / 'WGameCities.csv'
WDSTAGE1_WNCAA_COMPACT_RESULTS = WDATA_STAGE1 / 'WNCAATourneyCompactResults.csv'
WDSTAGE1_WNCAA_DETAILED_RESULTS = WDATA_STAGE1 / 'WNCAATourneyDetailedResults.csv'
WDSTAGE1_WNCAA_SEEDS = WDATA_STAGE1 / 'WNCAATourneySeeds.csv'
WDSTAGE1_WNCAA_SLOTS = WDATA_STAGE1 / 'WNCAATourneySlots.csv'
WDSTAGE1_WREGULAR_COMPACT_RESULTS = WDATA_STAGE1 / 'WRegularSeasonCompactResults.csv'
WDSTAGE1_WREGULAR_DETAILED_RESULTS = WDATA_STAGE1 / 'WRegularSeasonDetailedResults.csv'
WDSTAGE1_SEASONS = WDATA_STAGE1 / 'WSeasons.csv'
WDSTAGE1_TEAM_CONFERENCES = WDATA_STAGE1 / 'WTeamConferences.csv'
WDSTAGE1_TEAMS = WDATA_STAGE1 / 'WTeams.csv'
WDSTAGE1_TEAM_SPELLINGS = WDATA_STAGE1 / 'WTeamSpellings.csv'

NEW_NAME_DICT = {
    'Albany NY': 'SUNY Albany',
    'Santa Barbara': 'UC Santa Barbara',
    'VA Commonwealth': 'VCU',
    'Edwardsville': 'SIUE',
    'Cal Poly SLO': 'Cal Poly',
    'IPFW': 'PFW',
    'Long Island': 'LIU Brooklyn',
    'ULL': 'Louisiana'
}

# MODEL SETTINGS

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators=200,
                                                    n_jobs=-1,
                                                    verbose=2),
    'extratress': ensemble.ExtraTreesClassifier(n_estimators=200,
                                                n_jobs=-1,
                                                verbose=2)
}

