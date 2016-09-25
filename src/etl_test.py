# encoding: utf-8
import os
import logging
import pandas
import sys

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(APP_ROOT)
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'
from protos.feature import *
from protos.feature_orig import LIST_COLUMN_NUM
from protos.feature_zero import LIST_COLUMN_CAT_ZERO, LIST_COLUMN_NUM_ZERO
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def etl(test_data, num, feature_column, date_cols):

    logger.info('load end')
    d_cols = [col + "_DUR" for col in date_cols]

    test_data[d_cols] = test_data[date_cols]
    test_data[d_cols] -= test_data[date_cols].min(axis=1)
    logger.info('date end')

    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        test_data['part_L%s' % i] = test_data[cols].apply(lambda row: 1 if max(row) < 0 else 0, axis=1)
    logger.info('part end')

    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        col_names = ['L%s_D_MIN' % i, 'L%s_D_AVG' % i, 'L%s_D_MAX' % i, 'L%s_D_DIFF' % i]
        test_data[col_names[0]] = test_data[cols].min(axis=1)
        test_data[col_names[1]] = test_data[cols].mean(axis=1)
        test_data[col_names[2]] = test_data[cols].max(axis=1)
        test_data[col_names[3]] = test_data[col_names[2]] - test_data[col_names[0]]
        logger.info('line date %s end' % i)
    num_column = [col for col in feature_column if col in LIST_COLUMN_NUM]
    for i in list(range(4)) + ['']:
        cols = [col for col in num_column if 'L%s' % i in col]
        test_data['L%s_NUM_MAX' % i] = test_data[cols].max(axis=1)
        test_data['L%s_NUM_MIN' % i] = test_data[cols].min(axis=1)
        test_data['L%s_NUM_AVG' % i] = test_data[cols].mean(axis=1)

    df = test_data[['Id'] +
                   feature_column]
    logger.info('size %s %s' % df.shape)
    df.to_csv('../data/test_etl2/test_elt_%s.csv' % num, index=False)

if __name__ == '__main__':

    feature_column = [col for col in LIST_FEATURE_COLUMN_NAME
                      if col not in LIST_DUPLIDATE_CAT]
    feature_column = [col for col in feature_column
                      if col not in LIST_DUPLIDATE_DATE]
    feature_column = [col for col in feature_column
                      if col not in LIST_SAME_COL]
    feature_column = [col for col in feature_column
                      if col not in LIST_COLUMN_CAT_ZERO]
    feature_column = [col for col in feature_column
                      if col not in LIST_COLUMN_NUM_ZERO]
    origina_cols = list(feature_column)
    date_cols = [col for col in feature_column if 'D' in col]
    feature_column += [col + "_DUR" for col in date_cols]

    for i in range(4):
        feature_column.append('part_L%s' % i)

    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        col_names = ['L%s_D_MIN' % i, 'L%s_D_AVG' % i, 'L%s_D_MAX' % i, 'L%s_D_DIFF' % i]
        feature_column += col_names
        logger.info('line date %s end' % i)
    for i in list(range(4)) + ['']:
        feature_column += ['L%s_NUM_MAX' % i, 'L%s_NUM_MIN' % i, 'L%s_NUM_AVG' % i]

    test_data_all = pandas.read_csv(TEST_DATA, usecols=origina_cols + ['Id'], chunksize=10000)
    num = 0
    for test_data in test_data_all:
        etl(test_data, num, feature_column, date_cols)
        num += 1
