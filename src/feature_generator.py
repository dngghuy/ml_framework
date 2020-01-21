"""
This module is responsible for the feature transformation, feature generation, etc.
I tried to map each specified feature with its corresponding feature processing pipeline.
Pros: Scalable, easy to control and debug
Cons: Long script
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer # TODO: Replace FillNA with Imputer
from . import dispatcher


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


class CustomizedlBinarizer(TransformerMixin):
    """
    This class is the customized version of sklearn LabelBinarizer.
    Just to fit the LabelBinarizer with the Pipeline
    """
    def __init__(self, yes_no_dict={}):
        self.mapping_dict = yes_no_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transformed_df = X.apply(lambda x: x.map(self.mapping_dict), axis=0)
        return np.array(transformed_df, dtype=int)


def fill_na_mean(df, to_int=False):
    """
    Fill NA by the mean of the remaining values
    """
    val_to_field = df[~df.iloc[:, 0].isna()].mean().item()
    df = df.fillna(val_to_field)
    if to_int:
        df = df.astype(np.uint8)

    return df


def fill_na_min(df, to_int=False):
    """
    Fill NA by the min of the remaining values
    """
    val_to_field = df[~df.iloc[:, 0].isna()].min().item()
    df = df.fillna(val_to_field)
    if to_int:
        df = df.astype(np.uint8)

    return df


def fill_na_max(df, to_int=False):
    """
    Fill NA by the max of the remaining values
    """
    val_to_field = df[~df.iloc[:, 0].isna()].max().item()
    df = df.fillna(val_to_field)
    if to_int:
        df = df.astype(np.uint8)

    return df


class FillNA(TransformerMixin):
    style_dict = {
        'mean': fill_na_mean,
        'max': fill_na_max,
        'min': fill_na_min,
    }

    def __init__(self, style='mean', to_int=False):
        assert style in self.style_dict, f"The define style should be in {list(self.style_dict.keys())}"
        self.style = style
        self.to_int = to_int
        self.style_callable = self.style_dict.get(style)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.style_callable(X, self.to_int)