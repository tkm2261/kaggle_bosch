# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from utils import mcc_optimize, evalmcc_xgb_min
from feature import LIST_FEATURE_COLUMN_NAME
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
    """
    s = (true_pos + false_neg) / n
    p = (true_pos + false_pos) / n

    a = true_pos / n - s * p
    b = numpy.sqrt(s * p * (1 - s) * (1 - p))
    """
    a = true_pos * true_neg - false_pos * false_neg
    b = (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)
    return a / numpy.sqrt(b)


def mcc_scoring(estimator, X, y):

    y_pred_prb = estimator.predict_proba(X)[:, 1]
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
    return max_score

from numba.decorators import jit


@jit
def mcc_scoring2(y_pred_prb, y):
    list_thresh = numpy.arange(1, 100) / 100
    max_score = -1
    idx = None
    for thresh in list_thresh:
        y_pred = numpy.where(y_pred_prb >= thresh, 1, 0)
        score = mcc(y, y_pred)
        if score > max_score:
            max_score = score
            idx = thresh
    return idx, max_score


if __name__ == '__main__':
    logger.info('load start')
    target = pandas.read_csv('stack_1_target_2.csv')['0'].values
    data = pandas.read_csv('stack_1_data_2.csv').values
    logger.info('load end')
    logger.info('shape %s %s' % data.shape)
    logger.info('shape %s' % target.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), float(sum(target)) / target.shape[0]))

    params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.3,
              'max_depth': 3, 'min_child_weight': 0.01, 'n_estimators': 200,
              'scale_pos_weight': 10}

    # 4/8 param: {'learning_rate': 0.1, 'colsample_bytree': 1, 'scale_pos_weight': 1, 'n_estimators': 100, 'subsample': 1, 'min_child_weight': 1, 'max_depth': 4}
    # 2016-09-27/15:59:07 __main__ 132 [INFO][<module>] thresh:
    # 0.225158065557, total score: 0.264650750521, max_score: 0.264650750521

    all_params = {'max_depth': [3, 5, 7],
                  'n_estimators': [200, 100],
                  'learning_rate': [0.1],
                  'min_child_weight': [1, 0.5],
                  'subsample': [1, 0.7],
                  'reg_alpha': [0.1],
                  'colsample_bytree': [1, 0.8],
                  'scale_pos_weight': [1]}
    _all_params = {'C': [10**i for i in range(-3, 2)],
                   'penalty': ['l2']}
    cv = StratifiedKFold(target, n_folds=10, shuffle=True, random_state=0)
    list_score = []
    max_score = -100
    best_thresh = None
    pg = list(ParameterGrid(all_params))
    for i in range(data.shape[1]):
        thresh, score = mcc_optimize(data[:, i], target)
        logger.info('model:%s, thresh: %s, total score: %s, max_score: %s' % (i, thresh, score, max_score))

    for i, params in enumerate(pg):
        logger.info('%s/%s param: %s' % (i + 1, len(pg), params))
        pred_proba_all = []
        y_true = []
        for train_idx, test_idx in cv:
            model = XGBClassifier(seed=0)
            #model = LogisticRegression(n_jobs=-1, class_weight='balanced')
            model.set_params(**params)

            model.fit(data[train_idx], target[train_idx],
                      eval_metric=evalmcc_xgb_min,
                      verbose=False)

            #pred_proba = data[test_idx, -1]
            pred_proba = model.predict_proba(data[test_idx])[:, 1]
            pred_proba_all = numpy.r_[pred_proba_all, pred_proba]
            y_true = numpy.r_[y_true, target[test_idx]]
            score = roc_auc_score(target[test_idx], pred_proba)
            #logger.info('    score: %s' % score)
            #thresh, score = mcc_scoring(model, data[test_idx], target[test_idx])
            list_score.append(score)
            #logger.info('    thresh: %s' % thresh)
        score = numpy.mean(list_score)
        thresh, score = mcc_optimize(pred_proba_all, y_true)
        max_score = max(max_score, score)
        logger.info('thresh: %s, total score: %s, max_score: %s' % (thresh, score, max_score))
        if max_score == score:
            best_param = params
            best_thresh = thresh
    logger.info('best_thresh: %s, total max score: %s' % (best_thresh, max_score))
    model = XGBClassifier(seed=0)
    #model = LogisticRegression(n_jobs=-1, class_weight='balanced')
    model.set_params(**best_param)
    model.fit(data[train_idx], target[train_idx],
              eval_metric=evalmcc_xgb_min,
              verbose=False)

    with open('stack_model_2.pkl', 'wb') as f:
        pickle.dump(model, f, -1)
