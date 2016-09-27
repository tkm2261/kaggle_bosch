# encoding: utf-8
import os
import logging
import numpy
import pandas
import sys

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(APP_ROOT)
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join_pos.csv.gz')
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


def etl(train_data, feature_column):

    logger.info('load end')
    df = pandas.DataFrame()
    df['Id'] = train_data['Id']
    df[TARGET_COLUMN_NAME] = train_data[TARGET_COLUMN_NAME]
    df['hash'] = train_data[feature_column].astype(str).sum(axis=1).apply(
        hash)  # .apply(lambda x: ''.join(map(str, x)), axis=1)
    logger.info('size %s %s' % df.shape)

    return df

if __name__ == '__main__':

    train_data_all = pandas.read_csv(TRAIN_DATA, chunksize=10000)
    feature_column = [col for col in LIST_FEATURE_COLUMN_NAME]

    df = pandas.concat(etl(train_data, feature_column) for train_data in train_data_all)
    df = df.groupby('hash').count().sort_values('Id')
    df['cluster'] = numpy.arange(df.shape[0])
    df[df['Id'] > 10].to_csv('hash.csv')
