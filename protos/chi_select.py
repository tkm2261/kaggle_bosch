# encoding: utf-8
import os
import sys
import logging
import pandas
import pickle
import numpy
import glob
import hashlib
import gc

from multiprocessing import Pool
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, log_loss
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')
sys.path.append(APP_ROOT)

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from utils import mcc_optimize, evalmcc_xgb_min
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE

from stack.feature_1019 import LIST_ZERO_STACK_2, LIST_ZERO_STACK_2_2
from stack.feature_1021 import LIST_ZERO_STACK_3

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def make_chi():
    logger.info('CHI!!')
    data = pandas.read_csv('../protos/train_chi_all_3000.csv.gz')
    new_cols = ['L_chi_%s' % col if col != 'Id' else col for col in data.columns.values]
    data.columns = new_cols
    feature_columns = [col for col in new_cols if col not in LIST_ZERO_STACK_2 +
                       LIST_ZERO_STACK_2_2 + LIST_ZERO_STACK_3]

    data[feature_columns].to_csv(os.path.join(DATA_DIR, 'train_chi_3000.csv.gz'), index=False, compression='gzip')


def make_chi_test():
    logger.info('CHI2!!')
    data = pandas.read_csv('../protos/test_chi_all_3000.csv.gz')
    new_cols = ['L_chi_%s' % col if col != 'Id' else col for col in data.columns.values]
    data.columns = new_cols
    feature_columns = [col for col in new_cols if col not in LIST_ZERO_STACK_2 +
                       LIST_ZERO_STACK_2_2 + LIST_ZERO_STACK_3]
    data[feature_columns].to_csv(os.path.join(DATA_DIR, 'test_chi_3000.csv.gz'), index=False, compression='gzip')


if __name__ == '__main__':
    logger.info('load start')
    make_chi()
    make_chi_test()
