import pandas as pd
from . import dispatcher
from sklearn import model_selection


def default_train_valid_split(input_df, numfold):
    """
    Defining the default train-validation splitting.
    The default style is Stratified k-fold
    """
    print('*** Using Stratified K-Fold')
    kf = model_selection.StratifiedKFold(n_splits=numfold,
                                         shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=input_df, y=input_df['target'].values)):
        print(f'Len train: {len(train_idx)} \t Len val: {len(val_idx)}')
        input_df.loc[val_idx, 'kfold'] = fold
    input_df.to_csv(dispatcher.DATA_FOLDER / 'default_training_valid_set.csv', index=False)


def adversarial_train_valid_split():
    pass


class CreatingFolds:
    """
    This class is responsible for creating the train-validation splitting.
    Currently, I am using 2 types of making validation set, one is Stratified (default)
    and the other is Adversarial Validation.
    """
    style_dict = {
        'default': default_train_valid_split,
        'adversarial': adversarial_train_valid_split,
    }
    def __init__(self, numfolds, style='default'):
        """
        :param numfolds: int: Number of splits
        :param style: str: The desired splitting type.
        """
        self.numfolds = numfolds
        assert style in self.style_dict, f"The input 'style' must be in {list(self.style_dict.keys())}"
        self.style = style
        self.style_callable = self.style_dict.get(style)

    def __call__(self, input_df):
        """
        Make split here
        """
        assert isinstance(input_df, pd.DataFrame), f"The input should be pandas.DataFrame, here {type(input_df)}"
        input_df['kfold'] = -1
        input_df = input_df.sample(frac=1, random_state=7664).reset_index(drop=True)
        print('Making validation')
        self.style_callable(input_df, self.numfolds)


if __name__ == '__main__':
    train_df = pd.read_csv(dispatcher.INIT_TRAIN_CSV)
    train_df_cols = list(train_df.columns)
    # Check if the target column is mentioned by the new or old name
    train_df.rename(columns=dispatcher.CHANGE_NAME_DICT,
                    inplace=True)
    create_folds = CreatingFolds(numfolds=dispatcher.NUM_FOLDS,
                                 style='default')
    create_folds(train_df)