"""
This module handles the some preprocessing for categorical features.
This module will be called by feature_generator.
"""
import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


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






