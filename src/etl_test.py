# encoding: utf-8
import os
import logging
import pandas
import sys
import glob
import re
import gc

from multiprocessing import Pool
APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(APP_ROOT)
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'
from protos.feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE, LIST_SAME_COL
from protos.feature_orig import LIST_COLUMN_NUM
from protos.feature_zero import LIST_COLUMN_CAT_ZERO, LIST_COLUMN_NUM_ZERO
from protos.feature_0925 import LIST_COLUMN_ZERO

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def etl(train_data, num, feature_column, date_cols):

    logger.info('load end')
    logger.info('size %s %s' % train_data.shape)
    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        tmp = train_data[cols]
        train_data['part_L%s' % i] = tmp.apply(lambda row: 1 if max(row) < 0 else 0, axis=1)
    logger.info('part end')

    logger.info('size %s %s' % train_data.shape)
    for i in list(range(4)) + ['']:
        cols = [col for col in date_cols if 'L%s' % i in col]
        col_names = ['L%s_D_MIN' % i, 'L%s_D_AVG' % i, 'L%s_D_MAX' % i, 'L%s_D_DIFF' % i]
        tmp = train_data[cols]
        train_data[col_names[0]] = tmp.min(axis=1)
        train_data[col_names[1]] = tmp.mean(axis=1)
        train_data[col_names[2]] = tmp.max(axis=1)
        train_data[col_names[3]] = train_data[col_names[2]] - train_data[col_names[0]]
        logger.info('line date %s end' % i)

    logger.info('size %s %s' % train_data.shape)
    num_column = [col for col in LIST_FEATURE_COLUMN_NAME if col in LIST_COLUMN_NUM]
    for i in list(range(4)) + ['']:
        cols = [col for col in num_column if 'L%s' % i in col]
        tmp = train_data[cols]
        train_data['L%s_NUM_MAX' % i] = tmp.max(axis=1)
        train_data['L%s_NUM_MIN' % i] = tmp.min(axis=1)
        train_data['L%s_NUM_AVG' % i] = tmp.mean(axis=1)
        logger.info('line num %s end' % i)

    logger.info('size %s %s' % train_data.shape)
    for i in range(52):
        cols = [col for col in num_column if 'S%s' % i in col]
        if len(cols) == 0:
            continue
        line = cols[0][1]
        tmp = train_data[cols]
        train_data['L%s_S%s_NUM_MAX' % (line, i)] = tmp.max(axis=1)
        train_data['L%s_S%s_NUM_MIN' % (line, i)] = tmp.min(axis=1)
        train_data['L%s_S%s_NUM_AVG' % (line, i)] = tmp.mean(axis=1)
        logger.info('line num sec %s end' % i)
        logger.info('size %s %s' % train_data.shape)
    df = train_data[['Id'] +
                    feature_column]
    df['hash'] = df[feature_column].apply(lambda x: hash(''.join(map(str, x))), axis=1)
    logger.info('size %s %s' % df.shape)
    df.to_csv('../data/test_etl/test_elt_%s.csv' % num, index=False)

if __name__ == '__main__':

    feature_column = [col for col in LIST_FEATURE_COLUMN_NAME
                      if col not in LIST_DUPLIDATE_CAT]
    feature_column = [col for col in feature_column
                      if col not in LIST_DUPLIDATE_DATE]
    feature_column = [col for col in feature_column
                      if col not in LIST_SAME_COL]

    date_cols = [col for col in feature_column if 'D' in col]

    feature_column = [col for col in feature_column
                      if col not in LIST_COLUMN_CAT_ZERO]
    feature_column = [col for col in feature_column
                      if col not in LIST_COLUMN_NUM_ZERO]

    for i in range(4):
        feature_column.append('part_L%s' % i)

    for i in list(range(4)) + ['']:
        cols = [col for col in date_cols if 'L%s' % i in col]
        col_names = ['L%s_D_MIN' % i, 'L%s_D_AVG' % i, 'L%s_D_MAX' % i, 'L%s_D_DIFF' % i]
        feature_column += col_names
        logger.info('line date %s end' % i)

    for i in list(range(4)) + ['']:
        feature_column += ['L%s_NUM_MAX' % i, 'L%s_NUM_MIN' % i, 'L%s_NUM_AVG' % i]

    num_column = [col for col in LIST_FEATURE_COLUMN_NAME if col in LIST_COLUMN_NUM]
    for i in range(52):
        cols = [col for col in num_column if 'S%s' % i in col]
        if len(cols) == 0:
            continue
        line = cols[0][1]
        col_names = ['L%s_S%s_NUM_MAX' % (line, i), 'L%s_S%s_NUM_MIN' % (line, i), 'L%s_S%s_NUM_AVG' % (line, i)]
        feature_column += col_names

    feature_column = [col for col in feature_column
                      if col not in LIST_COLUMN_ZERO]
    feature_column.append('hash')
    path = sys.argv[1]
    train_data_all = pandas.read_csv(path, chunksize=10000)
    num = 0
    file_num = re.match(u'.*_(\d+).csv.gz$', path).group(1)
    for train_data in train_data_all:
        postfix = '%s_%s' % (file_num, num)
        etl(train_data, postfix, feature_column, date_cols)
        num += 1
        del train_data
        gc.collect()
