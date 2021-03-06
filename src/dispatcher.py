import pathlib
from sklearn import ensemble

# ### CONSTANT SETTINGS ###
# DATA PATH SETTINGS
DATA_PATH = pathlib.Path(__file__).parent.parent / 'data'


# PER-PRJ SETTINGS
# CAT IN THE DAT II
# TRAIN, TEST, OTHER DATA PATH SETTINGS
SAMPLE_TRAIN_PATH = DATA_PATH / 'cat_in_the_dat' / 'train.csv'
SAMPLE_TEST_PATH = DATA_PATH / 'cat_in_the_dat' / 'test.csv'
SAMPLE_SUB_PATH = DATA_PATH / 'cat_in_the_dat' / 'sample_submission.csv'

# BACKORDER
# TRAIN, TEST
BACKORDER_TRAIN_PATH = DATA_PATH / 'backorder' / 'training_set.csv'
BACKORDER_TEST_PATH = DATA_PATH / 'backorder' / 'test_set_sample.csv'

# MODEL SETTINGS

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators=200,
                                                    n_jobs=-1,
                                                    verbose=2),
    'extratress': ensemble.ExtraTreesClassifier(n_estimators=200,
                                                n_jobs=-1,
                                                verbose=2)
}

