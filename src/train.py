import pandas as pd
import numpy as np
import joblib
import argparse
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from . import feature_generator
from . import dispatcher
from . import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False)
parser.add_argument('--fep', required=False)
parser.add_argument('--grid_search', required=False)
args = parser.parse_args()


utils.check_make_dirs(dispatcher.VISUALIZE_FOLDER)
utils.check_make_dirs(dispatcher.MODELS_FOLDER)


TRAINING_DATA = dispatcher.DEFAULT_TRAIN_CSV
NUM_FOLDS = dispatcher.NUM_FOLDS

if args.model:
    MODEL = args.model
    assert MODEL in dispatcher.MODELS, "The mentioned model key should appear in dispatcher.MODELS."
else:
    print('No model specified, running default RandomForest in dispatcher.')
    MODEL = 'randomforest'

if args.fep:
    FEATURE_SET = args.fep
    assert FEATURE_SET in feature_generator.FEATURES, "The mentioned FEP key should appear in feature_generator.FEATURES"
else:
    print('No feature engineer process mentioned, running the naivest one.')
    FEATURE_SET = 'fep_naive'

if args.grid_search:
    GRID_SEARCH = args.grid_search.lower()
    assert GRID_SEARCH in ['true', 'false'], "Can only set grid search to True or False."
else:
    print('Set GRID SEARCH False by default.')
    GRID_SEARCH = 'false'


FOLD_MAPPING = utils.create_fold_mapping_dict(NUM_FOLDS)


class SimpleTrainWithFEP:
    def __init__(self, model, feature_engineer_pipeline=None, target_pipeline=None):
        self.feature_engineer_pipeline = feature_engineer_pipeline
        self.model = model
        self.target_pipeline = target_pipeline

    def fit(self, X_train, y_train):
        if self.feature_engineer_pipeline:
            X_train = self.feature_engineer_pipeline.fit_transform(X_train)

        if self.target_pipeline:
            y_train = self.feature_engineer_pipeline.fit_transform(y_train)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test, use_fep=True):
        if (self.feature_engineer_pipeline is not None) & (use_fep):
            X_test = self.feature_engineer_pipeline.fit_transform(X_test)
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        try:
            return self.model.predict_proba(X_test)
        except AttributeError:
            print("The current model does not have 'predict_proba' method")
            return None

    def predict_log_pproba(self, X_test):
        try:
            return self.model.predict_log_proba(X_test)
        except AttributeError:
            print("The current model does not have 'predict_log_proba' method")
            return None


def plotting_ROC_curves(list_y_true, list_y_pred_prob, saved_name=None):
    fig = plt.figure(figsize=(15, 10))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for f, (y_true, y_pred_prob) in enumerate(zip(list_y_true, list_y_pred_prob)):
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC={roc_auc} at fold {f}')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc = 'lower right')
    fig.savefig(dispatcher.VISUALIZE_FOLDER / saved_name)


class TrainWithFEP_CV:
    def __init__(self, model_info, additional_params={}, feature_engineer_pipeline_info=None, target_pipeline_info=None):
        self.feature_engineer_pipeline_name, self.feature_engineer_pipeline = feature_engineer_pipeline_info
        self.model = model_info
        self.target_pipeline_name, self.target_pipeline = target_pipeline_info
        self.additional_params = additional_params

    def _run_cv_one(self, df, fold):
        train_df = df[df['kfold'].isin(FOLD_MAPPING.get(fold))]
        valid_df = df[df['kfold'] == fold]

        y_train = train_df[['target']]
        y_valid = valid_df[['target']]

        if self.feature_engineer_pipeline:
            train_df = self.feature_engineer_pipeline.fit_transform(train_df)
            valid_df = self.feature_engineer_pipeline.fit_transform(valid_df)
            joblib.dump(self.feature_engineer_pipeline, f"{dispatcher.MODELS_FOLDER}/{self.model}_{self.feature_engineer_pipeline_name}_{fold}_fep.pkl")
        if self.target_pipeline:
            y_train = self.target_pipeline.fit_transform(y_train)
            y_valid = self.target_pipeline.fit_transform(y_valid)

            y_train = np.ravel(y_train)
            y_valid = np.ravel(y_valid)
            joblib.dump(target_process, f"{dispatcher.MODELS_FOLDER}/{self.model}_{self.target_pipeline_name}_{fold}_tp.pkl")

        trained_model = dispatcher.MODELS[self.model]
        trained_model_params = trained_model.get_params()
        trained_model_params.update(self.additional_params)
        trained_model.set_params(**trained_model_params)
        trained_model.fit(train_df, y_train)
        val_preds = trained_model.predict_proba(valid_df)[:, 1]

        # Save the model (Since dependencies are transformed into df already)
        joblib.dump(trained_model, f"models/{self.model}_{fold}.pkl")

        return trained_model, val_preds, y_valid

    def run_cv(self, df, with_visualize_roc=False):
        trained_models = []
        list_val_preds = []
        list_y_valid = []
        num_fold = len(np.unique(df['kfold'].values))
        for fold in tqdm(range(num_fold)):
            trained_model, val_preds, y_valid = self._run_cv_one(df, fold)
            trained_models.append(trained_model)
            list_val_preds.append(val_preds)
            list_y_valid.append(y_valid)
        if with_visualize_roc:
            saved = f'ROC_Curve_{self.model}_{self.feature_engineer_pipeline_name}.png'
            plotting_ROC_curves(list_y_valid, list_val_preds, saved)

        return trained_models


class SimpleGridSearchPipelineBinaryClf:
    def __init__(self, model_info, feature_engineer_pipeline_info=None, target_pipeline_info=None):
        self.feature_engineer_pipeline_name, self.feature_engineer_pipeline = feature_engineer_pipeline_info
        self.target_pipeline_name, self.target_pipeline = target_pipeline_info
        self.model_name = model_info
        # Set pipeline for model right here, based on dispatcher file.
        self.model = dispatcher.MODELS[self.model_name]
        self.model_pipeline = Pipeline([
            (f'{self.model_name}', self.model)
        ])
        self.model_params_grid = {'__'.join([self.model_name, key]): val
                                  for key, val in dispatcher.MODELS_GRID_PARAMS[self.model_name].items()}

    def run_grid_search(self, df):
        Y = df[['target']]
        if self.target_pipeline:
            Y = self.target_pipeline.fit_transform(Y)

        if self.feature_engineer_pipeline:
            df = self.feature_engineer_pipeline.fit_transform(df)

        searchCV = GridSearchCV(estimator=self.model_pipeline,
                                param_grid=self.model_params_grid,
                                cv=dispatcher.NUM_FOLDS,
                                error_score='f1')
        searchCV.fit(df, Y)
        self._collecting_best_result(searchCV)

        return searchCV

    def _collecting_best_result(self, searchCV):
        best_params_dict = searchCV.best_params_
        joblib.dump(best_params_dict, f"{dispatcher.MODELS_FOLDER}/{self.model_name}_{self.feature_engineer_pipeline_name}_bestParams.pkl")

def simple_train_one_fold_clf_wfep(model, df, fold, feature_process=None, target_process=None):
    train_df = df[df['kfold'].isin(FOLD_MAPPING.get(fold))]
    valid_df = df[df['kfold']==fold]

    y_train = train_df[['target']]
    y_valid = valid_df[['target']]

    if feature_process:
        train_df = feature_process.fit_transform(train_df)
        valid_df = feature_process.fit_transform(valid_df)
        joblib.dump(feature_process, f"models/{model}_{fold}_fep.pkl")
    if target_process:
        y_train = target_process.fit_transform(y_train)
        y_valid = target_process.fit_transform(y_valid)
        joblib.dump(target_process, f"models/{model}_{fold}_tp.pkl")

    trained_model = dispatcher.MODELS[model]
    trained_model.fit(train_df, y_train)
    val_preds = trained_model.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(y_valid, val_preds))

    # Save the model (Since dependencies are transformed into df already)
    joblib.dump(trained_model, f"models/{model}_{fold}.pkl")

    return trained_model


if __name__ == '__main__':
    """
    Training process should start like this:
    Feature processing -> Grid search -> Best params -> Take best params to run CV -> save models and feature pipelines
    """
    df = pd.read_csv(TRAINING_DATA)
    feature_process, target_process = feature_generator.FEATURES[FEATURE_SET]
    if GRID_SEARCH == 'true':
        simple_grid_search = SimpleGridSearchPipelineBinaryClf(model_info=MODEL,
                                                               feature_engineer_pipeline_info=feature_process,
                                                               target_pipeline_info=target_process)
        results = simple_grid_search.run_grid_search(df=df)
        grid_search_best_params = results.best_params_
    else:
        grid_search_best_params = {}

    training = TrainWithFEP_CV(model_info=MODEL,
                               additional_params=grid_search_best_params,
                               feature_engineer_pipeline_info=feature_process,
                               target_pipeline_info=target_process)
    training.run_cv(df, with_visualize_roc=True)
