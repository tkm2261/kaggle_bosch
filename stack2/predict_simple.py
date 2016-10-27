# encoding: utf-8

import pickle
import os
import pandas
import logging
import numpy
import gc
import glob
from multiprocessing import Pool
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM

from train_feature_1 import LIST_TRAIN_COL
from train_feature_2 import LIST_TRAIN_COL as LIST_TRAIN_COL2

from feature_1009 import LIST_COLUMN_ZERO_MIX
from feature_1026 import LIST_ZERO_COL, LIST_ZERO_COL2, LIST_ZERO_COL3
from feature_1026_cnt import LIST_ZERO_COL_CNT, LIST_ZERO_COL_CNT2
from feature_1026_all import LIST_ZERO_COL_ALL

from feature_1026_2 import LIST_ZERO_COL
from feature_1026_2_cnt import LIST_ZERO_COL_CNT
from feature_1026_2_all import LIST_ZERO_COL_ALL, LIST_ZERO_COL_ALL2, LIST_ZERO_COL_ALL3

from xgboost import DMatrix

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TRAIN_POSITIVE_DATA = os.path.join(DATA_DIR, 'train_simple_join_pos.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')
logger = logging.getLogger(__name__)


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    df = pandas.read_csv(filename)
    return df


def make_chi(df, feature_columns):
    logger.info('CHI!!')

    data = pandas.read_csv('../data/test_chi_1000.csv.gz')
    feature_columns += [col for col in data.columns.values if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns


def make_chi2(df, feature_columns):
    logger.info('CHI!!')
    """
    data = pandas.read_csv('../protos/train_chi_all_1000.csv.gz')
    new_cols = ['L_chi_%s' % col for col in data.columns.values]
    data.columns = new_cols
    """

    data = pandas.read_csv('../data/test_chi_2000.csv.gz')
    feature_columns += [col for col in data.columns.values if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns


def make_chi3(df, feature_columns):
    logger.info('CHI!!')
    data = pandas.read_csv('../data/test_chi_3000.csv.gz')
    feature_columns += [col for col in data.columns.values if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns


def main():

    with open('xgb_model_1.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('xgb_model_2.pkl', 'rb') as f:
        model2 = pickle.load(f)

    p = Pool()

    test_data = pandas.concat(p.map(read_csv,
                                    glob.glob(os.path.join(DATA_DIR, 'test_join/*'))
                                    )).reset_index(drop=True)
    p.close()
    p.join()

    logger.info('end load')

    gc.collect()
    logger.info('end merge')
    from train_feature_2_1 import LIST_TRAIN_COL
    feature_column = LIST_TRAIN_COL
    data = test_data[feature_column].fillna(-10)
    pred = []

    cnt = 0
    train_dmatrix = DMatrix(data, label=None)
    del data
    gc.collect()

    for j, jj in enumerate(['']):
        cols = [col for col in feature_column if 'L%s' % jj in col]
        logger.info('%s' % (model.__repr__()))
        pred = model.predict(train_dmatrix)

    pred = pandas.DataFrame(pred,
                            columns=['L_pred'],
                            index=test_data['Id'].values)

    logger.info('end pred1')
    df = test_data.merge(pred, how='left', left_on='Id', right_index=True, copy=False)
    from train_feature_2_2 import LIST_TRAIN_COL
    feature_column = LIST_TRAIN_COL

    data = df[feature_column].fillna(-10)
    logger.info('end 1')
    pred = []
    train_dmatrix = DMatrix(data, label=None)
    del data
    gc.collect()
    for j, jj in enumerate([0, 1, 2, 3, '']):
        cols = [col for col in feature_column if 'L%s' % jj in col]
        logger.info('%s' % (model2.__repr__()))
        pred = model2.predict(train_dmatrix)

    predict_proba = pred
    predict_proba2 = pred.mean(axis=1)

    logger.info('end pred2')

    predict = numpy.where(predict_proba >= 0.224, 1, 0)
    logger.info('end predict')
    ans = pandas.DataFrame(df['Id'])
    ans['Response'] = predict
    ans['proba'] = predict_proba
    ans['proba2'] = predict_proba2
    for m in range(pred.shape[1]):
        ans['m%s' % m] = pred[:, m]

    ans.to_csv('submit.csv', index=False)

if __name__ == '__main__':
    main()
