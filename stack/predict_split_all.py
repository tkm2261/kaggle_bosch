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


def read_df(df):
    return df


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


def make_chi(df, feature_columns):
    logger.info('CHI!!')
    """
    data = pandas.read_csv('../protos/train_chi_all_1000.csv.gz')
    new_cols = ['L_chi_%s' % col for col in data.columns.values]
    data.columns = new_cols
    """

    data = pandas.read_csv('../data/train_chi_1000.csv.gz')
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

    data = pandas.read_csv('../data/train_chi_2000.csv.gz')
    feature_columns += [col for col in data.columns.values if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns


def make_chi3(df, feature_columns):
    logger.info('CHI!!')
    data = pandas.read_csv('../data/train_chi_3000.csv.gz')
    feature_columns += [col for col in data.columns.values if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns


def main():

    logger.info('start load')

    feature_column = LIST_TRAIN_COL
    with open('list_xgb_model.pkl', 'rb') as f:
        list_model = pickle.load(f)
    with open('list_xgb_model_2.pkl', 'rb') as f:
        list_model2 = pickle.load(f)

    with open('stack_model_2.pkl', 'rb') as f:
        fin_model = pickle.load(f)
    """
    p = Pool()

    df = pandas.concat(p.map(read_csv,
                             glob.glob(os.path.join(DATA_DIR, 'test_join/*'))
                             )).reset_index(drop=True)

    p.close()
    p.join()
    logger.info('end load %s %s' % df.shape)
    gc.collect()
    logger.info('end merge')

    feature_column_1 = [col for col in feature_column if 'L_pred' not in col]
    data = df[feature_column_1].fillna(-10)
    pred = []

    cnt = 0
    for j, jj in enumerate([0, 1, 2, 3, '']):
        cols = [col for col in feature_column_1 if 'L%s' % jj in col]
        for s in range(5):
            model = list_model[cnt]
            logger.info('(%s, %s)' % (j, s))
            logger.info('%s' % (model.__repr__()))
            pred.append(model.predict_proba(data[cols]))
            cnt += 1

    pred = pandas.DataFrame(numpy.array(pred).T,
                            columns=['L0_L1_L2_L3_pred_%s' % col for col in range(cnt)],
                            index=df['Id'].values)
    pred.to_csv('ans_stack_1.csv.gz', compression='gzip')
    logger.info('end pred1')
    df = df.merge(pred, how='left', left_on='Id', right_index=True, copy=False)
    df, feature_column = make_chi(df, feature_column)
    df, feature_column = make_chi2(df, feature_column)
    df, feature_column = make_chi3(df, feature_column)

    del data
    gc.collect()
    data = df[LIST_TRAIN_COL2].fillna(-10)
    logger.info('end 1')
    pred = []

    cnt = 0
    for j, jj in enumerate([0, 1, 2, 3, '']):
        cols = [col for col in LIST_TRAIN_COL2 if 'L%s' % jj in col]
        for s in range(5):
            logger.info('(%s, %s)' % (j, s))
            logger.info('%s' % (model.__repr__()))
            model = list_model2[cnt]
            pred.append(model.predict_proba(data[cols]))
            cnt += 1

    pred = pandas.DataFrame(numpy.array(pred).T,
                            columns=['L0_L1_L2_L3_pred2_%s' % col for col in range(cnt)],
                            index=df['Id'].values)

    pred.to_csv('ans_stack_2.csv.gz', compression='gzip')
    """
    pred = pandas.read_csv('ans_stack_2.csv.gz', index_col='Id')
    logger.info('last: %s' % (fin_model.__repr__()))
    predict_proba = fin_model.predict_proba(pred.values)[:, 1]
    predict_proba2 = pred.mean(axis=1)

    logger.info('end pred2')

    predict = numpy.where(predict_proba >= 0.224, 1, 0)
    logger.info('end predict')
    ans = pandas.DataFrame(pred.index.values)
    ans.columns = ['Id']
    ans['Response'] = predict
    ans['proba'] = predict_proba
    ans['proba2'] = predict_proba2
    for m in range(pred.shape[1]):
        ans['m%s' % m] = pred.values[:, m]

    ans.to_csv('submit.csv', index=False)

if __name__ == '__main__':
    main()
