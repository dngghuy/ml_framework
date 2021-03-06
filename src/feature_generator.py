"""
This module is responsible for the feature transformation, feature generation, etc.
I tried to map each specified feature with its corresponding feature processing pipeline.
Pros: Scalable, easy to control and debug
Cons: Long script
"""
from .basic_import import *
from . import dispatcher
from . import categorical
from . import numerical


class Selector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    From this good tutorial https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
    """
    def __init__(self, key):
        if not isinstance(key, list):
            self.key = [key]
        else:
            self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


if __name__ == '__main__':
    import pandas as pd
    import pathlib
    from . import dispatcher
    ## Using cat_in_the_dat data for testing ##
    sample_path = pathlib.Path(__file__).parent.parent / 'data' / 'cat_in_the_dat' / 'train.csv'
    sample_df = pd.read_csv(sample_path)
    sample_train = pd.read_csv(dispatcher.SAMPLE_TRAIN_PATH)
    ## Test fill NAs
    # sample_train_cols = list(sample_train.columns)
    # sample_numerical_train_select = ['bin_0', 'bin_1', 'bin_2']
    # sample_categorical_train_select = ['ord_5']
    # sample_fill_pipeline = Pipeline([
    #     ('selector', Selector(key=sample_numerical_train_select)),
    #     ('fill_max', numerical.FillNA(style='max', to_int=np.uint8))
    # ])
    # sample_fill_cat_pipeline = Pipeline([
    #     ('selector', Selector(key=sample_categorical_train_select)),
    #     ('fill_Huy', categorical.FillNAStr(fill_str='Huy'))
    # ])
    # sample_other_select = [i for i in sample_train_cols if i not in sample_numerical_train_select + sample_categorical_train_select]
    # sample_selects = Pipeline([
    #     ('selector', Selector(key=sample_other_select))
    # ])
    #
    # feat_unions = FeatureUnion([
    #     ('fill_nan', sample_fill_pipeline),
    #     ('fill_nan_str', sample_fill_cat_pipeline),
    #     ('only_select', sample_selects)
    # ])
    # feature_process = Pipeline([
    #     ('feats', feat_unions),
    # ])
    # print(sample_other_select)
    # new_sample = feature_process.fit_transform(sample_train)

    ## Test label encoder
    sample_lbl_encoder = Pipeline([
        ('selector', Selector(key=['ord_2', 'ord_3'])),
        ('fill_cat', categorical.FillNAStr(fill_str='NA')),
        ('lbl_encdr', categorical.CategoricalLabelEncoding())
    ])
    new_sample = sample_lbl_encoder.fit_transform(sample_train)

    print('#### After preprocessing')
    new_sample_df = pd.DataFrame(new_sample)
    new_sample_df.columns = ['ord_2', 'ord_3']
    print(new_sample_df.head())
    print(sample_lbl_encoder.named_steps['lbl_encdr'].label_encoders['ord_3'].classes_)
    # new_sample_df.columns = sample_numerical_train_select + sample_categorical_train_select + sample_other_select
    #

    ## Using backorder data for testing ##
    # train_csv = pd.read_csv(dispatcher.BACKORDER_TRAIN_PATH)
    # print(train_csv.head(10))
    # binary_cols_v2 = Pipeline([
    #     ('selector', Selector(key=['ppap_risk', 'rev_stop'])),
    #     ('binarizer', categorical.CustomizedlBinarizer(yes_no_dict={'Yes': 1, 'No': 0}))
    # ])
    # train_csv_tmp = binary_cols_v2.fit_transform(train_csv)
    # print(train_csv_tmp)



