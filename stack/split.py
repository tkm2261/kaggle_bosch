# encoding: utf-8
import os
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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from utils import mcc_optimize, evalmcc_xgb_min
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM
from feature_0928 import LIST_COLUMN_ZERO
from feature_1009 import LIST_COLUMN_ZERO_MIX
from feature_1015 import LIST_ZEOO_2
from omit_pos_id import LIST_OMIT_POS_ID, LIST_OMIT_POS_MIN

from train_feature_1 import LIST_TRAIN_COL

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    df = pandas.read_csv(filename)
    return df


def read_df(df, i):
    df.to_csv(os.path.join(DATA_DIR, 'train_join/train_join_%s.csv.gz' % i), index=False, compression='gzip')
    logger.info('load end %s' % i)

if __name__ == '__main__':
    logger.info('load start')
    p = Pool()

    train_data = pandas.concat(p.map(read_csv,
                                     glob.glob(os.path.join(DATA_DIR, 'train_etl/*'))
                                     ), ignore_index=True)

    train_data_cnt = pandas.concat(p.map(read_csv,
                                         glob.glob(os.path.join(DATA_DIR, 'train_etl2/*'))
                                         ), ignore_index=True)

    p.close()
    p.join()
    logger.info('end load')
    logger.info('a %s %s' % train_data.shape)
    logger.info('cnt %s %s' % train_data_cnt.shape)
    feature_column = [col for col in train_data.columns.values if col not in LIST_COLUMN_ZERO_MIX +
                      [TARGET_COLUMN_NAME, 'Id']]
    feature_column_cnt = [
        col for col in train_data_cnt.columns.values if col not in LIST_COLUMN_ZERO_MIX + [TARGET_COLUMN_NAME, 'Id']]

    train_data = train_data[[TARGET_COLUMN_NAME, 'Id'] + feature_column]
    train_data_cnt = train_data_cnt[['Id'] + feature_column_cnt]

    gc.collect()
    data = train_data.merge(train_data_cnt, how='left', left_on='Id', right_on='Id', copy=False)
    logger.info('end merge')
    del train_data
    del train_data_cnt
    gc.collect()

    i = 0
    idx = data.index.values
    num = 10000
    res = []
    p = Pool()
    for i in range(int(len(idx) / num) + 1):
        if i * num > len(idx):
            break

        if (i + 1) * num > len(idx):
            ix = idx[i * num:]
        else:
            ix = idx[i * num: (i + 1) * num]

        df = data.ix[ix]
        res.append(p.apply_async(read_df, (df, i)))
        i += 1
    [r.get() for r in res]
    p.close()
    p.join()
