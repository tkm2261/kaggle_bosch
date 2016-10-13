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


if __name__ == '__main__':
    logger.info('load start')
    target = pandas.read_csv('stack_1_target_1.csv')['0'].values
    data = pandas.read_csv('stack_1_data_1.csv').values
    logger.info('load end')
    logger.info('shape %s %s' % data.shape)
    logger.info('shape %s' % target.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), float(sum(target)) / target.shape[0]))

    with open('list_xgb_model.pkl', 'rb') as f:
        list_model = pickle.load(f)

    for i in range(data.shape[1]):
        thresh, score = mcc_optimize(data[:, i], target)
        auc_score = roc_auc_score(target, data[:, i])
        print('"%s"' % list_model[i].__repr__(), auc_score, thresh, score, sep=',')
