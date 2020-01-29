"""
This module is responsible for the feature transformation, feature generation, etc.
I tried to map each specified feature with its corresponding feature processing pipeline.
Pros: Scalable, easy to control and debug
Cons: Long script
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Imputer # TODO: Replace FillNA with Imputer
from . import dispatcher
# from . import categorical


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


def fill_na_str(df, fill_str='NA'):
    """
    Fill NA by some defined string
    """
    df = df.fillna(fill_str)

    return df


class FillNAStr(TransformerMixin):
    def __init__(self, fill_str='NA'):
        self.fill_str = str(fill_str)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return fill_na_str(X, self.fill_str)



if __name__ == '__main__':
    import pandas as pd
    import pathlib
    sample_path = pathlib.Path(__file__).parent.parent / 'data' / 'cat_in_the_dat' / 'train.csv'
    sample_df = pd.read_csv(sample_path)
    #feature_cols = [i for i in list(sample_df.columns) if i not in ['id', 'target']]
    print(sample_df)
    print(sample_df.isna().any())
    print(sample_df.dtypes)

    bin_4 = Pipeline([
        ('selector', Selector(key='bin_4')),
        ('fillna_str', FillNAStr(fill_str='NA'))
    ])

    sample_df['bin_4'] = bin_4.fit_transform(sample_df)
    print(sample_df)
    print(sample_df.isna().any())
