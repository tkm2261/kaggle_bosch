# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob
import hashlib
import gc
import gzip

from multiprocessing import Pool
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from tffm import TFFMClassifier
from pyfm import pylibfm

from sklearn.preprocessing import OneHotEncoder

from fastFM.sgd import FMClassification
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
from omit_pos_id import LIST_OMIT_POS_ID, LIST_OMIT_POS_MIN
import tensorflow as tf
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def hash_groupby(df, feature_column):

    new_feature_column = list(feature_column)
    for col in feature_column:
        if 'hash' not in col:
            continue
        logger.info('hash col%s' % col)
        tmp = df.groupby(col)[[TARGET_COLUMN_NAME]].count()
        # tmp_mean = tmp[TARGET_COLUMN_NAME].mean()
        # tmp[TARGET_COLUMN_NAME][tmp[TARGET_COLUMN_NAME] < 2] = 2
        tmp.columns = [col + '_prob']
        new_feature_column.append(col + '_prob')
        df = pandas.merge(df, tmp, how='left', left_on=col, right_index=True)

    df[[col for col in new_feature_column if 'hash' in col]].to_csv('hash_prob.csv', index=True)
    return df, new_feature_column


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    df = pandas.read_csv(filename)
    return df


def make_cross(df, feature_columns):
    mst = pandas.read_csv('cross_term.csv', header=None, index_col=0)[1]
    mst = mst[mst > 500]
    for pair in mst.index.values:
        f1, f2 = pair.split('-')
        df[pair] = df[f1] * df[f2]
        feature_columns.append(pair)
        logger.info('cross: %s' % pair)
    return df, feature_columns


def make_stack(df, feature_columns):
    logger.info('STACKING!!')
    ids = pandas.read_csv('stack_1_id_1.csv')['0'].values
    data = pandas.read_csv('stack_1_data_1.csv')

    new_cols = ['L_pred_%s' % col for col in data.columns.values]
    data.columns = new_cols
    feature_columns += new_cols

    data['Id'] = ids
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')
    return df, feature_columns


def read_df(df):
    return df

if __name__ == '__main__':
    logger.info('load start')
    p = Pool()

    all_data = pandas.concat(p.map(read_csv,
                                   glob.glob(os.path.join(DATA_DIR, 'train_rank/*'))
                                   )).reset_index(drop=True)
    p.close()
    p.join()
    all_data = all_data.fillna(0)
    enc = OneHotEncoder(dtype=numpy.int32, handle_unknown='ignore')

    target = all_data[TARGET_COLUMN_NAME].values
    #target = numpy.where(target > 0, 1, -1)

    ids = all_data['Id'].values
    all_data = all_data[[col for col in all_data.columns.values if col != TARGET_COLUMN_NAME and col != 'Id']]
    data = enc.fit_transform(all_data)
    del all_data
    gc.collect()

    from sklearn.datasets import dump_svmlight_file

    logger.info('load end')
    pos_rate = float(sum(target)) / target.shape[0]
    logger.info('shape %s %s' % data.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), pos_rate))

    all_params = {
        'num_factors': [5],
        'num_iter': [10],
    }

    cv = StratifiedKFold(target, n_folds=3, shuffle=True, random_state=0)
    all_ans = None
    all_target = None
    all_ids = None
    from io import BytesIO
    logger.info('cv_start')
    for params in ParameterGrid(all_params):
        logger.info('param: %s' % (params))
        for train_idx, test_idx in list(cv)[:1]:
            with gzip.open('train_fm.svm', 'wb') as f:
                dump_svmlight_file(data[train_idx], target[train_idx], f)
            del output
            gc.collect()
            with gzip.open('test_svm.svm', 'wb') as f:
                dump_svmlight_file(data[test_idx], target[test_idx], f)

            model = TFFMClassifier(order=2,
                                   rank=10,
                                   optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                                   n_epochs=50,
                                   batch_size=100000,
                                   init_std=0.001,
                                   reg=0.001,
                                   input_type='sparse'
                                   )
            """
            model = FMClassification()
            """
            model.fit(data[train_idx], target[train_idx], show_progress=True)
            ans = model.predict_proba(data[test_idx])[:, 1]

            score = roc_auc_score(target[test_idx], ans)
            logger.info('score: %s' % score)
            logger.info('all thresh: %s, score: %s' % mcc_optimize(ans, target[test_idx]))
            score = roc_auc_score(target[test_idx], ans)
