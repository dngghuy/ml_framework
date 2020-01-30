"""
This module handles the some preprocessing for categorical features.
This module will be called by feature_generator.
"""
from .basic_import import np, pd, TransformerMixin, preprocessing, deepcopy, pathlib, joblib


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
        if X.shape[1] == 1:
            return fill_na_str(X, self.fill_str)
        else:
            X_cols = list(X.columns)
            for col in X_cols:
                X[[col]] = fill_na_str(X[[col]], self.fill_str)

            return X


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


class CategoricalLabelEncoding(TransformerMixin):
    """
    Replicate and modify a bit from Abhishek Thakur, in order to fit into pipeline
    """
    def __init__(self, existing_lbl=None):
        if existing_lbl is not None:
            if isinstance(existing_lbl, str) or isinstance(existing_lbl, pathlib.Path):
                self.label_encoders = joblib.load(existing_lbl)
            else:
                self.label_encoders = existing_lbl
                self.exist_lbl = True
        else:
            self.label_encoders = dict()
            self.exist_lbl = False

    def fit(self, X, y=None):
        return self

    def _label_encoding(self, df_val):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df_val + ['ZUnknown'])
        lbl_transform = lbl.transform(df_val)

        return lbl_transform, lbl

    def _label_encoding_w_encoder(self, encoder, df_val):
        all_unique_val = np.unique(df_val)
        all_new_item = []
        for unique_item in all_unique_val:
            if unique_item not in encoder.classes_:
                all_new_item.append(unique_item)
        new_df_val = ['ZUnknown' if x in all_new_item else x for x in df_val]
        lbl_transform = encoder.transform(new_df_val)

        return lbl_transform

    def transform(self, X, y=None):
        cols = list(X.columns)
        X_output = deepcopy(X)
        if not self.exist_lbl:
            for col in cols:
                col_transform, lbl = self._label_encoding(list(X[col].values))
                X_output.loc[:, col] = col_transform
                self.label_encoders[col] = lbl
        else:
            for col in cols:
                lbl = self.label_encoders[col]
                X_output.loc[:, col] = self._label_encoding_w_encoder(encoder=lbl,
                                                                      df_val=list(X[col].values))
        return X_output


class CategoricalLabelBinarization(TransformerMixin):
    def __init__(self, existing_lbl=None):
        if existing_lbl is not None:
            if isinstance(existing_lbl, str) or isinstance(existing_lbl, pathlib.Path):
                self.label_encoders = joblib.load(existing_lbl)
            else:
                self.label_encoders = existing_lbl
                self.exist_lbl = True
        else:
            self.label_encoders = dict()
            self.exist_lbl = False

    def fit(self, X, y=None):
        return self

    def _label_binarization(self, colname, df_values):
        lbl = preprocessing.LabelBinarizer()
        lbl.fit(df_values)
        val = lbl.transform(df_values)
        new_cols = [''.join([colname, f"__bin_{i}"]) for i in range(val.shape[1])]
        val_df = pd.DataFrame(val)
        val_df.columns = new_cols

        return val_df, lbl

    def _label_binarization_w_binarizer(self, binarizer, colname, df_values):
        val = binarizer.transform(df_values)
        new_cols = [''.join([colname, f"__bin_{i}"]) for i in range(val.shape[1])]
        val_df = pd.DataFrame(val)
        val_df.columns = new_cols

        return val_df

    def transform(self, X, y=None):
        cols = list(X.columns)
        X_output = deepcopy(X)
        if not self.exist_lbl:
            for col in cols:
                col_transform, lbl = self._label_binarization(colname=col,
                                                              df_values=list(X[col].values))
                X_output = pd.concat([X_output, col_transform], axis=1)
                self.label_encoders[col] = lbl
        else:
            for col in cols:
                lbl = self.label_encoders[col]
                col_transform = self._label_binarization_w_binarizer(binarizer=lbl,
                                                                     colname=col,
                                                                     df_values=list(X[col].values))
                X_output = pd.concat([X_output, col_transform], axis=1)

        return X_output
