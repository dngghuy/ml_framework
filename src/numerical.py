"""
This module handles the some preprocessing for numerical features.
This module will be called by feature_generator.
"""
from .basic_import import np, TransformerMixin


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