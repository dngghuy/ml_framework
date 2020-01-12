from sklearn import ensemble
0.75091

MODELS = {
    'randomforest': ensemble.RandomForestClassifier(n_estimators=200,
                                                    n_jobs=-1,
                                                    verbose=2),
    'extratress': ensemble.ExtraTreesClassifier(n_estimators=200,
                                                n_jobs=-1,
                                                verbose=2)
}

# import pathlib
# from sklearn import ensemble
#
# # DATA PATH
# DATA_FOLDER = pathlib.Path(__file__).parent.parent / 'data'
# INIT_TRAIN_CSV = DATA_FOLDER / 'training_set.csv'
# DEFAULT_TRAIN_CSV = DATA_FOLDER / 'default_training_valid_set.csv'
# TEST_CSV = DATA_FOLDER / 'test_set_sample.csv'
# SAMPLE_CSV = DATA_FOLDER / 'test_set_sample_predictions.csv'
#
#
# # RE-FORMAT THINGS
# CHANGE_NAME_DICT = {'went_on_backorder': 'target',
#                     'sku': 'id'}
#
#
# # TRAIN-VALIDATION RELATED
# NUM_FOLDS = 5
#
# # MODELS RELATED
# MODELS = {
#     'randomforest': ensemble.RandomForestClassifier(n_estimators=200,
#                                                     n_jobs=-1,
#                                                     verbose=2),
#     'extratress': ensemble.ExtraTreesClassifier(n_estimators=200,
#                                                 n_jobs=-1,
#                                                 verbose=2)
# }
