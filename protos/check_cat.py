# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from utils import mcc_optimize, evalmcc_xgb_min
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM, LIST_COLUMN_CAT

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('load start')
    # train_data = pandas.read_csv(TRAIN_DATA)
    train_data = pandas.concat(pandas.read_csv(path, usecols=['Id', 'Response'] + LIST_COLUMN_CAT) for path in glob.glob(
        os.path.join(DATA_DIR, 'train_simple_part/*'))[:1]).reset_index(drop=True)
    print(train_data.groupby(LIST_COLUMN_CAT).agg({'Id': 'count', 'Response': 'sum'}))
