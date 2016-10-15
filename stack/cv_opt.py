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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.fixes import bincount, parallel_helper

from scipy.optimize import minimize, fmin
from multiprocessing import Pool
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


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


class TookTooLong(Warning):
    pass

import time
import warnings


class MinimizeStopper(object):

    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()

    def __call__(self, xk=None):
        print('hogehoge')
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
        else:
            # you might want to report other stuff here
            print("Elapsed: %.3f sec" % elapsed)


class NMOpt:

    def __init__(self):

        self.w_ = None

    def fit(self, X, y, seed=0, X_test=None, y_test=None):
        numpy.random.seed(seed)
        # w_ = numpy.random.random(X.shape[1])
        # model = LogisticRegression(n_jobs=-1, random_state=0)
        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, min_samples_leaf=5, random_state=0)
        model.fit(X, y)
        w_ = numpy.ones(n_estimators)
        X = model._validate_X_predict(X)

        def func(x):
            W = Parallel(n_jobs=-1, verbose=0,
                         backend="threading")(
                delayed(parallel_helper)(e, 'predict_proba', X,
                                         check_input=False)
                for e in model.estimators_)
            W = numpy.array([w[:, 1] for w in W]).T
            # W = numpy.array([m.predict_proba(X)[:, 1] for m in model.estimators_]).T
            score = numpy.dot(W, x)
            score = score / score.max()  # sigmoid(score)
            thresh, score = mcc_optimize(score, y)
            if 1:  # numpy.random.random() < 0.1:
                logger.info('  thresh: %s, score: %s' % (thresh, score))

            return - score

        def ttt(xk):
            print('hoge')
        res = minimize(func, w_, method='Powell', callback=ttt)
        self.w_ = res.x
        self.model = model

    def predict_proba(self, X):
        X = self.model._validate_X_predict(X)
        W = Parallel(n_jobs=-1, verbose=0,
                     backend="threading")(
            delayed(parallel_helper)(e, 'predict_proba', X,
                                     check_input=False)
            for e in self.model.estimators_)
        W = numpy.array([w[:, 1] for w in W]).T
        score = numpy.dot(W, self.w_)
        score = score / score.max()  # sigmoid(score)

        return numpy.array([1 - score, score]).T


def make_data():
    ids = pandas.read_csv('stack_1_id_2.csv')['0'].values
    target = pandas.read_csv('stack_1_target_2.csv')['0'].values
    data = pandas.read_csv('stack_1_data_2.csv')
    data['Id'] = ids
    data[TARGET_COLUMN_NAME] = target
    """
    logger.info('shape %s %s' % data.shape)
    ids1 = pandas.read_csv('stack_1_id_1.csv')['0'].values
    data1 = pandas.read_csv('stack_1_data_1.csv')
    data1['Id'] = ids1
    logger.info('shape %s %s' % data1.shape)

    data = data.merge(data1, left_on='Id', right_on='Id', copy=False)
    """
    return data

if __name__ == '__main__':
    logger.info('load start')
    df = make_data()
    data = df[[col for col in df.columns.values if col != 'Id' and col != TARGET_COLUMN_NAME]].values
    target = df[TARGET_COLUMN_NAME].values
    logger.info('load end')
    logger.info('shape %s %s' % data.shape)
    logger.info('shape %s' % target.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), float(sum(target)) / target.shape[0]))

    cv = StratifiedKFold(target, n_folds=5, shuffle=True, random_state=0)
    list_score = []
    max_score = -100
    best_thresh = None
    pg = list(ParameterGrid({'a': [0]}))
    # for i in range(data.shape[1]):
    #    thresh, score = mcc_optimize(data[:, i], target)
    #    logger.info('model:%s, thresh: %s, total score: %s, max_score: %s' % (i, thresh, score, max_score))

    for i, params in enumerate(pg):
        logger.info('%s/%s param: %s' % (i + 1, len(pg), params))
        pred_proba_all = []
        y_true = []
        for train_idx, test_idx in cv:
            model = NMOpt()

            model.fit(data[train_idx], target[train_idx])

            # pred_proba = data[test_idx, -1]
            pred_proba = model.predict_proba(data[test_idx])[:, 1]
            pred_proba_all = numpy.r_[pred_proba_all, pred_proba]
            y_true = numpy.r_[y_true, target[test_idx]]
            score = roc_auc_score(target[test_idx], pred_proba)
            # logger.info('    score: %s' % score)
            # thresh, score = mcc_scoring(model, data[test_idx], target[test_idx])
            list_score.append(score)
            # logger.info('    thresh: %s' % thresh)
        score = numpy.mean(list_score)
        thresh, score = mcc_optimize(pred_proba_all, y_true)
        max_score = max(max_score, score)
        logger.info('thresh: %s, total score: %s, max_score: %s' % (thresh, score, max_score))
        if max_score == score:
            best_param = params
            best_thresh = thresh
    logger.info('best_thresh: %s, total max score: %s' % (best_thresh, max_score))
    # model = XGBClassifier(seed=0)
    # model = LogisticRegression(n_jobs=-1, class_weight='balanced')
    # model.set_params(**best_param)
    model = NMOpt()
    model.fit(data[train_idx], target[train_idx])

    with open('stack_model_2.pkl', 'wb') as f:
        pickle.dump(model, f, -1)
