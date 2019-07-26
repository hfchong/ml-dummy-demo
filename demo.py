"""Demo training script"""
import itertools
import logging
import os

from environs import Env
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from bedrock_client.bedrock.api import BedrockApi
from custom_pkg.modelling import CustomRF


FEATURES_PATH = os.path.join(os.getenv("STORAGE_PATH"),
                             os.getenv("FEATURES_FILE"))
LABELS_PATH = os.path.join(os.getenv("STORAGE_PATH"),
                           os.getenv("LABELS_FILE"))
OUTPUT_MODEL_PATH = os.path.join('/artefact',
                                 os.getenv('OUTPUT_MODEL_NAME'))

env = Env()
PARAMS_SPACE = {
    'n_estimators': env.list("N_ESTIMATORS_LIST", subcast=int),
    'max_depth': env.list("MAX_DEPTH_LIST", subcast=int),
    'min_samples_split': env.list("MIN_SAMPLES_SPLIT_LIST", subcast=int)
}


def load_data():
    """Loads the raw data"""
    ft = pd.read_csv(FEATURES_PATH)
    lb = pd.read_csv(LABELS_PATH)
    df = ft.merge(lb)
    df.pop('user')
    X, y = df, df.pop('label')

    return train_test_split(X, y, test_size=0.2)


def train():
    """Trains and saves the best model"""
    X_train, X_test, y_train, y_test = load_data()
    params_generator = (
        dict(zip(PARAMS_SPACE.keys(), v))
        for v in itertools.product(*PARAMS_SPACE.values())
    )

    best_model = {'roc_auc': 0.0,
                  'pr_auc': 0.0,
                  'params': None,
                  'model': None}
    for i, params in enumerate(params_generator):
        print('RUN: {}'.format(i + 1))
        print('PARAMS: {}'.format(params))
        clf = CustomRF(**params)
        # fit model
        clf.fit(X_train, y_train)

        # score test
        y_proba = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        if pr_auc > best_model['pr_auc']:
            best_model['params'] = params
            best_model['roc_auc'] = roc_auc
            best_model['pr_auc'] = pr_auc
            best_model['model'] = clf

    save_model_metrics(best_model)


def save_model_metrics(best_model):
    """Saves the best model and logs the relevant metrics"""
    joblib.dump(best_model['model'], OUTPUT_MODEL_PATH)

    logger = logging.getLogger(__name__)
    bedrock = BedrockApi(logger)
    for param, value in best_model['params'].items():
        bedrock.log_metric(param, value)
    bedrock.log_metric("ROC AUC", best_model['roc_auc'])
    bedrock.log_metric("PR AUC", best_model['pr_auc'])


if __name__ == '__main__':
    train()
