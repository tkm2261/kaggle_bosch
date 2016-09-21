# encoding: utf-8

import pickle
import os
import pandas
import logging
import numpy
import gc

from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME

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


def main():
    logger.info('start load')
    feature_column = LIST_FEATURE_COLUMN_NAME

    all_df = pandas.read_csv(TEST_DATA, compression='gzip', chunksize=100000, usecols=feature_column + ['Id'])

    logger.info('end load')
    store = pandas.HDFStore('predict.h5', complib='zlib')
    for i, df in enumerate(all_df):
        store.put(str(i), df, format='fixed')
        logger.info('chunk: %s' % i)
    store['size'] = i + 1
    store.close()

if __name__ == '__main__':
    # main()
    df = pandas.read_hdf('predict.h5', key=str(11))
    import pdb
    pdb.set_trace()
