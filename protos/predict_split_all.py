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


def hash_join(df):
    mst = pandas.read_csv('hash_prob.csv')
    for col in list(df.columns.values):
        if 'hash' not in col:
            continue
        logger.info('hash col%s' % col)
        tmp = mst.groupby(col)[[col + '_prob']].max()
        df = df.merge(tmp, how='left', left_on=col, right_index=True, copy=False)
        df[col + '_prob'] = df[col + '_prob'].fillna(1)

    return df


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    df = pandas.read_csv(filename)
    return df


def read_df(df):
    return df


def main():
    feature_column = LIST_TRAIN_COL
    with open('list_xgb_model.pkl', 'rb') as f:
        list_model2 = pickle.load(f)
    with open('list_xgb_model_1.pkl', 'rb') as f:
        list_model = pickle.load(f)

    with open('stack_model_1.pkl', 'rb') as f:
        fin_model = pickle.load(f)

    p = Pool()

    test_data = pandas.concat(p.map(read_csv,
                                    glob.glob(os.path.join(DATA_DIR, 'test_etl/*'))
                                    )).reset_index(drop=True)

    test_data_cnt = pandas.concat(p.map(read_df,
                                        pandas.read_csv(os.path.join(
                                            DATA_DIR, 'test_etl2/test.csv.gz'), chunksize=10000),
                                        )).reset_index(drop=True)
    p.close()
    p.join()
    logger.info('end load')
    df = test_data.merge(test_data_cnt, how='left', left_on='Id', right_on='Id', copy=False)
    del test_data
    del test_data_cnt
    gc.collect()
    logger.info('end merge')

    feature_column_1 = [col for col in feature_column if 'L_pred' not in col]
    data = df[feature_column_1].fillna(-10)
    pred = []

    cnt = 0
    for j, jj in enumerate([0, 1, 2, 3, '']):
        cols = [col for col in feature_column_1 if 'L%s' % jj in col]
        model = list_model[cnt]
        pred.append(model.predict_proba(data[cols])[:, 1])
        cnt += 1

    pred = pandas.DataFrame(numpy.array(pred).T,
                            columns=['L_pred_%s' % col for col in range(cnt)],
                            index=df['Id'].values)

    logger.info('end pred1')
    df = df.merge(pred, how='left', left_on='Id', right_index=True, copy=False)
    del data
    gc.collect()
    data = df[feature_column].fillna(-10)
    logger.info('end 1')
    pred = []

    cnt = 0
    for j, jj in enumerate([0, 1, 2, 3, '']):
        cols = [col for col in feature_column if 'L%s' % jj in col]
        model = list_model2[cnt]
        pred.append(model.predict_proba(data[cols])[:, 1])
        cnt += 1

    pred = numpy.array(pred).T

    predict_proba = fin_model.predict_proba(pred)[:, 1]
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
