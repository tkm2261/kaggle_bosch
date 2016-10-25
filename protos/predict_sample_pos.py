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

from pos_test import LIST_POS_TEST_ID


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    df = pandas.read_csv(filename)
    df = df[df['Id'].isin(LIST_POS_TEST_ID)]
    df[TARGET_COLUMN_NAME] = numpy.ones(df.shape[0], dtype=int)
    return df


def read_csv2(filename):
    'converts a filename to a pandas dataframe'
    mst = pandas.read_csv('test_neg_id.csv', header=None)
    mst = pandas.Series(mst[0].values, index=mst[0].values)

    df = pandas.read_csv(filename)
    df = df[df['Id'].isin(mst.index)]
    df[TARGET_COLUMN_NAME] = numpy.zeros(df.shape[0], dtype=int)
    return df


def read_df(df):
    return df


def out_df(df, i):
    df.to_csv(os.path.join(DATA_DIR, 'test_right/test_right_%s.csv.gz' % i), index=False, compression='gzip')
    logger.info('load end %s' % i)


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


def main():
    logger.info('start load')
    p = Pool()

    df = pandas.concat(p.map(read_csv,
                             glob.glob(os.path.join(DATA_DIR, 'test_join/*'))
                             )).reset_index(drop=True)
    p.close()
    p.join()
    p = Pool()
    print(df.shape)
    df2 = pandas.concat(p.map(read_csv2,
                              glob.glob(os.path.join(DATA_DIR, 'test_join/*'))
                              )).reset_index(drop=True)
    p.close()
    p.join()
    print(df2.shape)

    df = pandas.concat([df, df2], ignore_index=True)
    print(df.shape)
    num = 10000
    res = []
    p = Pool()
    idx = df.index.values
    for i in range(int(len(idx) / num) + 1):
        if i * num > len(idx):
            break

        if (i + 1) * num > len(idx):
            ix = idx[i * num:]
        else:
            ix = idx[i * num: (i + 1) * num]

        tmp = df.ix[ix]
        res.append(p.apply_async(out_df, (tmp, i)))
        i += 1
    [r.get() for r in res]

    p.close()
    p.join()


if __name__ == '__main__':
    main()
