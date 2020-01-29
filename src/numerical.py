"""
This module handles the some preprocessing for numerical features.
This module will be called by feature_generator.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def fill_na_mean(df, to_int=np.uint8, **kwargs):
    """
    Fill NA by the mean of the remaining values
    """
    val_to_field = df[~df.iloc[:, 0].isna()].mean().item()
    df = df.fillna(val_to_field)
    if to_int:
        df = df.astype(to_int)

    return df


def fill_na_min(df, to_int=np.uint8, **kwargs):
    """
    Fill NA by the min of the remaining values
    """
    val_to_field = df[~df.iloc[:, 0].isna()].min().item()
    df = df.fillna(val_to_field)
    if to_int:
        df = df.astype(to_int)

    return df


def fill_na_max(df, to_int=np.uint8, **kwargs):
    """
    Fill NA by the max of the remaining values
    """
    val_to_field = df[~df.iloc[:, 0].isna()].max().item()
    df = df.fillna(val_to_field)
    if to_int:
        df = df.astype(to_int)

    return df


def fill_na_num(df, to_int=np.uint8, num=-999):
    """
    Fill NA by some defined number
    """
    df = df.fillna(num)
    if to_int:
        df = df.astype(to_int)

    return df


class FillNA(TransformerMixin):
    style_dict = {
        'mean': fill_na_mean,
        'max': fill_na_max,
        'min': fill_na_min,
        'num': fill_na_num,
    }

    def __init__(self, style='mean', to_int=np.uint8, fill_num=None):
        assert style in self.style_dict, f"The define style should be in {list(self.style_dict.keys())}"
        self.style = style
        self.to_int = to_int
        self.num = fill_num
        if style=='num' and self.num is None:
            raise Exception("When style is 'num' then fill_num must not be None")
        self.style_callable = self.style_dict.get(style)
        self.style_callable_params = {'to_int': self.to_int,
                                      'num': self.num}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if X.shape[1] == 1:
            return self.style_callable(X, **self.style_callable_params)
        else:
            X_cols = list(X.columns)
            for col in X_cols:
                X[[col]] = self.style_callable(X[[col]], **self.style_callable_params)

            return X


if __name__ == '__main__':
    import pandas as pd
    from . import dispatcher
    from .feature_generator import Selector
    from sklearn.pipeline import Pipeline, FeatureUnion
    sample_train = pd.read_csv(dispatcher.SAMPLE_TRAIN_PATH)
    print(sample_train.head(10))
    print(sample_train.dtypes)
    print(sample_train.isna().any())
    sample_train_cols = list(sample_train.columns)
    sample_train_select = ['bin_0', 'bin_1', 'bin_2']
    sample_fill_pipeline = Pipeline([
        ('selector', Selector(key=sample_train_select)),
        ('fill_max', FillNA(style='max', to_int=np.uint8))
    ])
    sample_other_select = [i for i in sample_train_cols if i not in sample_train_select]
    sample_selects = Pipeline([
        ('selector', Selector(key=sample_other_select))
    ])

    feat_unions = FeatureUnion([
        ('fill_nan', sample_fill_pipeline),
        ('only_select', sample_selects)
    ])
    feature_process = Pipeline([
        ('feats', feat_unions),
    ])
    print(sample_other_select)
    new_sample = feature_process.fit_transform(sample_train)
    print('#### After preprocessing')
    new_sample_df = pd.DataFrame(new_sample)
    new_sample_df.columns = sample_train_select + sample_other_select

    print(new_sample_df.head(10))
    print(new_sample_df.dtypes)
    print(new_sample_df.isna().any())