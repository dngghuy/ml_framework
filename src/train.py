import pandas as pd
import numpy as np
import joblib
import argparse
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from . import feature_generator
from . import dispatcher
from . import utils
from . import cross_validation


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False)
parser.add_argument('--fep', required=False)
parser.add_argument('--grid_search', required=False)
args = parser.parse_args()


utils.check_make_dirs(dispatcher.VISUALIZE_FOLDER)
utils.check_make_dirs(dispatcher.MODELS_FOLDER)
utils.check_make_dirs(dispatcher.SUBMISSION_FOLDER)


# TRAINING_DATA = dispatcher.DEFAULT_TRAIN_CSV
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
    def __init__(self, model_info, version_name=None, additional_params={},
                 feature_engineer_pipeline_info=None, target_pipeline_info=None,
                 save_meta_feature=False, make_test_prediction=False):
        self.feature_engineer_pipeline_name, self.feature_engineer_pipeline = feature_engineer_pipeline_info
        self.model = model_info
        self.target_pipeline_name, self.target_pipeline = target_pipeline_info
        self.additional_params = additional_params
        if version_name is None:
            self.version_name = '_'.join([f'{model_info}', str(int(time.time()))])
        else:
            self.version_name = version_name
        self.save_folder_path = dispatcher.MODELS_FOLDER / self.version_name
        utils.check_make_dirs(self.save_folder_path)
        self.save_meta_feature = save_meta_feature
        self.make_test_prediction = make_test_prediction
        self.dict_meta = {}
        self.dict_test_preds = {}

    def _run_cv_one(self, df, fold, test_df):
        train_df = df[df['kfold'].isin(FOLD_MAPPING.get(fold))]
        valid_df = df[df['kfold'] == fold]
        train_df = train_df.drop(['kfold'], axis=1)
        valid_df = valid_df.drop(['kfold'], axis=1)

        y_train = train_df[['target']]
        y_valid = valid_df[['target']]

        if self.feature_engineer_pipeline:
            train_df = self.feature_engineer_pipeline.fit_transform(train_df)
            valid_df = self.feature_engineer_pipeline.fit_transform(valid_df)
            save_fep_name = f"{self.save_folder_path}/{self.model}_{self.feature_engineer_pipeline_name}_{fold}_fep.pkl"
            joblib.dump(self.feature_engineer_pipeline, save_fep_name)
        if self.target_pipeline:
            y_train = self.target_pipeline.fit_transform(y_train)
            y_valid = self.target_pipeline.fit_transform(y_valid)

            y_train = np.ravel(y_train)
            y_valid = np.ravel(y_valid)
            save_tp_name = f"{self.save_folder_path}/{self.model}_{self.target_pipeline_name}_{fold}_tp.pkl"
            joblib.dump(self.target_pipeline, save_tp_name)

        trained_model = dispatcher.MODELS[self.model]
        trained_model_params = trained_model.get_params()
        trained_model_params.update(self.additional_params)
        trained_model.set_params(**trained_model_params)
        trained_model.fit(train_df, y_train)
        val_preds = trained_model.predict_proba(valid_df)[:, 1]
        if self.save_meta_feature:
            self.dict_meta[fold] = val_preds
        if self.make_test_prediction:
            if self.feature_engineer_pipeline:
                test_df = self.feature_engineer_pipeline.fit_transform(test_df)
            test_preds = trained_model.predict_proba(test_df)[:, 1]
            self.dict_test_preds[fold] = test_preds

        # Save the model (Since dependencies are transformed into df already)
        save_model_name = f"{self.save_folder_path}/{self.model}_{fold}.pkl"
        joblib.dump(trained_model, save_model_name)

        return trained_model, val_preds, y_valid

    def run_cv(self, df, with_visualize_roc=False, test_df=None):
        if self.make_test_prediction and test_df is None:
            raise Exception('Should provide df_test if you want to make prediction also')
        trained_models = []
        list_val_preds = []
        list_y_valid = []
        num_fold = len(np.unique(df['kfold'].values))
        for fold in tqdm(range(num_fold)):
            trained_model, val_preds, y_valid = self._run_cv_one(df, fold, test_df=test_df)
            trained_models.append(trained_model)
            list_val_preds.append(val_preds)
            list_y_valid.append(y_valid)
        if with_visualize_roc:
            saved = f'ROC_Curve_{self.model}_{self.feature_engineer_pipeline_name}.png'
            plotting_ROC_curves(list_y_valid, list_val_preds, saved)
        if self.save_meta_feature:
            save_meta_name =  f"{self.save_folder_path}/{self.model}_all_meta_{num_fold}_folds.pkl"
            joblib.dump(self.dict_meta, save_meta_name)
        if self.make_test_prediction:
            # Here, we first assume the average
            all_predictions = list(self.dict_test_preds.values())
            # print(type(all_predictions))
            # print(all_predictions[0])
            predictions = np.sum(all_predictions, axis=0) / num_fold
        else:
            predictions = None

        return trained_models, predictions


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


if __name__ == '__main__':
    print('Hi')
    df = pd.read_csv(dispatcher.CUSTOM_BASELINE_FEATURE_CSV)
    test_df = pd.read_csv(dispatcher.CUSTOM_BASELINE_TEST_CSV)
    sample_cv = cross_validation.CrossValidation(df=df,
                                                 target_cols=['target'],
                                                 problem_type='binary_classification',
                                                 num_folds=5)
    new_df = sample_cv.split()
    print(new_df.shape)
    print(test_df.shape)
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
                               target_pipeline_info=target_process,
                               save_meta_feature=True,
                               make_test_prediction=True)
    trained_models, preds = training.run_cv(new_df, with_visualize_roc=True, test_df=test_df)
    submission_df = pd.read_csv(dispatcher.WSAMPLE_SUBMISSION)
    submission_df['Pred'] = preds
    submission_path = dispatcher.SUBMISSION_FOLDER / f'{training.version_name}.csv'
    submission_df.to_csv(submission_path, index=False)
