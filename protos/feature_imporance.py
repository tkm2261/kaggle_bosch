# encoding: utf-8

import pickle
import os
import pandas
import logging
import numpy
import gc

from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE

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
    with open('list_xgb_model.pkl', 'rb') as f:
        list_model = pickle.load(f)

    from train_feature_1 import LIST_TRAIN_COL
    feature_column = LIST_TRAIN_COL

    logger.info('end load %s' % len(list_model))
    logger.info('feature_num: %s %s' % (len(LIST_FEATURE_COLUMN_NAME), len(feature_column)))
    df_fi = pandas.DataFrame()
    df_fi['name'] = feature_column
    for j, jj in enumerate([0, 1, 2, 3, '']):
        cols = [col for col in feature_column if 'L%s' % jj in col]
        df_fi2 = pandas.DataFrame()
        df_fi2['name'] = cols
        df_fi2['model%s' % j] = list_model[j].feature_importances_
        df_fi = pandas.merge(df_fi, df_fi2, how='left', left_on='name', right_on='name')

    df_fi.sort_values('model4', ascending=False).to_csv('feature_importances.csv', index=False)
    return df_fi
if __name__ == '__main__':
    df = main()
