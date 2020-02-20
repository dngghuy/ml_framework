import pandas as pd
from sklearn import model_selection

"""
Supported CVs
- binary classification
- multiclass classification
- multi label classification
- single column regression
- multi column regression
- holdout
"""

class CrossValidation:
    def __init__(self, df,
                 target_cols=None,
                 shuffle=True,
                 problem_type='binary_classification',
                 multilabel_delimeter=',',
                 num_folds=5,
                 random_state=7664):
        """
        Initialize of cross validation obj
        :param df: pd.DataFrame: The target dataframe
        :param target_cols: str: The target columns, if None then 'target'
        :param shuffle: bool: Whether it should be shuffle or not
        :param problem_type: str: problem type
        :param multilabel_delimeter: str: delimeter for splitting labels
        :param num_folds: int: number of cvs
        :param random_state: int: random state
        """
        self.dataframe = df
        if target_cols is None:
            self.target_cols = ['target']
        else:
            self.target_cols = target_cols
        self.shuffle = shuffle
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.random_state = random_state
        self.multilabel_delimeter = multilabel_delimeter

        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

        self.dataframe['kfold'] = -1

    def split(self):
        print(f"Start making folds for the {self.problem_type} problem")
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            assert self.num_targets == 1, f"Invalid number of targets for this problem type: Expect 1, here {self.num_targets}"
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            assert unique_values != 1, f"Only one unique value found!"
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                 shuffle=(not self.shuffle))
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ['single_col_regression', 'multi_col_regression']:
            if self.num_targets != 1 and self.problem_type == 'single_col_regression':
                raise Exception(f"Invalid number of targets: With {self.problem_type}, num target should be 1, here {self.num_targets}")
            if self.num_targets < 2 and self.problem_type == 'multi_col_regression':
                raise Exception(f"Invalid number of targets: With {self.problem_type}, num target should be >= 2, here {self.num_targets}")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type == 'multilabel_classification':
            assert self.num_targets == 1, f"Invalid number of targets for this problem type: Expect 1, here {self.num_targets}"
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimeter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type.startswith("holdout_"):
            # Should shuffle before doing holdout:\
            if not self.shuffle:
                self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
            holdout_percentage = int(self.problem_type.split('_')[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        else:
            raise Exception("Problem type not yet defined")

        return self.dataframe




