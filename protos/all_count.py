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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'
from feature_1009 import LIST_COLUMN_ZERO_MIX


from utils import mcc_optimize, evalmcc_xgb_min
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM
from feature_0928 import LIST_COLUMN_ZERO
from omit_pos_id import LIST_OMIT_POS_ID, LIST_OMIT_POS_MIN

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    df = pandas.read_csv(filename)
    return df


def make_count(pair):
    name, series = pair
    logger.info(name)
    if name == 'Id' or name == TARGET_COLUMN_NAME:
        return series

    col_name = name + '_CNT'
    if col_name in LIST_COLUMN_ZERO_MIX:
        return None

    df_hash = pandas.DataFrame(series.values, columns=[col_name])
    df_cnt = df_hash.groupby(col_name)[[col_name]].count()
    df_hash.columns = ['hoge']
    df = df_hash.merge(df_cnt, how='left', left_on='hoge', right_index=True, copy=False)

    return df[col_name].fillna(1).astype(numpy.int32)


def save(data):
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


def read_df(df, i):
    df.to_csv(os.path.join(DATA_DIR, 'train_etl2/train_etl2_%s.csv.gz' % i), index=False, compression='gzip')
    logger.info('load end %s' % i)


def read_df2(df, i):
    df.to_csv(os.path.join(DATA_DIR, 'test_etl2/train_etl2_%s.csv.gz' % i), index=False, compression='gzip')
    logger.info('load end %s' % i)


def save_test(data):
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
        res.append(p.apply_async(read_df2, (df, i)))
        i += 1
    [r.get() for r in res]
    p.close()
    p.join()


if __name__ == '__main__':
    logger.info('load start')
    # train_data = pandas.read_csv(TRAIN_DATA)

    # train_data = pandas.concat(pandas.read_csv(path) for path in glob.glob(
    #    os.path.join(DATA_DIR, 'train_etl/*'))).reset_index(drop=True)
    p = Pool()
    train_data = pandas.concat(p.map(read_csv,
                                     glob.glob(os.path.join(DATA_DIR, 'train_etl/*')) +
                                     glob.glob(os.path.join(DATA_DIR, 'test_etl/*'))
                                     ), ignore_index=True)
    p.close()
    p.join()

    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']
    #feature_column = [col for col in feature_column if col not in LIST_COLUMN_ZERO]
    train_data = train_data[['Id', TARGET_COLUMN_NAME] + feature_column]
    logger.info('load end')
    p = Pool()
    list_series = p.map(make_count, train_data.iteritems())
    p.close()
    p.join()
    del train_data
    gc.collect()

    logger.info('conv end')
    aaa = pandas.DataFrame()
    for i, series in enumerate(list_series):
        if series is None:
            continue
        aaa[series.name] = series.values
        del series
        if i % 100 == 0:
            logger.info('pros %s' % i)
            gc.collect()

    logger.info('df end')
    save(aaa[aaa[TARGET_COLUMN_NAME] == aaa[TARGET_COLUMN_NAME]])
    save_test(aaa[aaa[TARGET_COLUMN_NAME] != aaa[TARGET_COLUMN_NAME]])
