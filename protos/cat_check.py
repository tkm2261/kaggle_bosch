# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_CAT
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def mcc(y_true, y_pred):
    n = y_true.shape[0]
    true_pos = sum(1. for i in range(n) if y_true[i] == 1 and y_pred[i] == 1)
    true_neg = sum(1. for i in range(n) if y_true[i] == 0 and y_pred[i] == 0)
    false_pos = sum(1. for i in range(n) if y_true[i] == 0 and y_pred[i] == 1)
    false_neg = sum(1. for i in range(n) if y_true[i] == 1 and y_pred[i] == 0)

    a = true_pos * true_neg - false_pos * false_neg
    b = (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)
    return a / numpy.sqrt(b)


def mcc_scoring(estimator, X, y):

    y_pred_prb = estimator.predict_proba(X)[:, 1]
    list_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    max_score = -1
    for thresh in list_thresh:
        y_pred = numpy.where(y_pred_prb >= thresh, 1, 0)
        score = mcc(y, y_pred)
        logger.info('thresh: %s, score: %s' % (thresh, score))
        if score > max_score:
            max_score = score
    return max_score


def mcc_scoring2(y_pred_prb, y):
    list_thresh = numpy.arange(1, 100) / 100
    max_score = -1
    idx = None
    for thresh in list_thresh:
        y_pred = numpy.where(y_pred_prb >= thresh, 1, 0)
        score = mcc(y, y_pred)
        logger.debug('thresh: %s, score: %s' % (thresh, score))
        if score > max_score:
            max_score = score
            idx = thresh
    return idx, max_score


def min_date(row):
    try:
        return row - min(ele for ele in row if ele > 0)
    except ValueError:
        return row


def date_stats(row, col_names):
    row_na = row[row > 0]
    if len(row_na) == 0:
        return pandas.Series([-10, -10, -10, -10], index=col_names)
    else:
        r_min = row_na.min()
        r_mean = row_na.mean()
        r_max = row_na.max()
        return pandas.Series([r_min, r_mean, r_max, r_max - r_min], index=col_names)

if __name__ == '__main__':
    logger.info('load start')
    train_data = pandas.read_csv(TRAIN_DATA)
    # with open('train_etl_sampling.pkl', 'rb') as f:
    #    train_data = pickle.load(f)
    logger.info('shape %s %s' % train_data.shape)
    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']

    params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.3,
              'max_depth': 5, 'min_child_weight': 0.01, 'n_estimators': 300,
              'scale_pos_weight': 1.}

    feature_column = [col for col in feature_column if col in LIST_COLUMN_CAT]

    logger.info('shape %s %s' % train_data.shape)
    target = train_data[TARGET_COLUMN_NAME].values.astype(numpy.bool_)
    data = train_data[feature_column].fillna(-10)
    cv = StratifiedKFold(target, n_folds=10, shuffle=True, random_state=0)
    logger.info('shape %s %s' % data.shape)
    logger.info('cv_start')
    if TARGET_COLUMN_NAME in feature_column:
        raise
    for train_idx, test_idx in list(cv)[:1]:

        model = XGBClassifier(seed=0)
        model.set_params(**params)
        model.fit(data.ix[train_idx], target[train_idx])

        pred = model.predict_proba(data.ix[test_idx])[:, 1]  # ans.max(axis=1)
        print(mcc_scoring2(pred, target[test_idx]))
        score = roc_auc_score(target[test_idx], pred)
        logger.info('score: %s' % score)

    df_fi = pandas.DataFrame()
    df_fi['name'] = feature_column
    df_fi['score'] = model.feature_importances_
    df_fi = df_fi.sort('score', ascending=False)
    df_fi.to_csv('feature_importances_cat.csv', index=False)
