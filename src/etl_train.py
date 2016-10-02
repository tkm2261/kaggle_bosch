# encoding: utf-8
import os
import logging
import pandas
import sys
import glob
import re
import gc
import hashlib
from multiprocessing import Pool
APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(APP_ROOT)
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'
from protos.feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE, LIST_SAME_COL
from protos.feature_orig import LIST_COLUMN_NUM, LIST_COLUMN_CAT, LIST_COLUMN_DATE
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
        for col in cols:
            train_data[col + '_%s_D'%i] = train_data[col] - train_data[col_names[0]]
        logger.info('line date %s end' % i)

    logger.info('size %s %s' % train_data.shape)
    num_column = [col for col in LIST_FEATURE_COLUMN_NAME if col in LIST_COLUMN_NUM]
    for i in list(range(4)) + ['']:
        cols = [col for col in num_column if 'L%s' % i in col]
        tmp = train_data[cols]
        train_data['L%s_NUM_MAX' % i] = tmp.max(axis=1)
        train_data['L%s_NUM_MIN' % i] = tmp.min(axis=1)
        train_data['L%s_NUM_AVG' % i] = tmp.mean(axis=1)
        train_data['L%s_NUM_DIFF' % i] = train_data['L%s_NUM_MAX' % i] - train_data['L%s_NUM_MIN' % i]
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
        train_data['L%s_S%s_NUM_DIFF' % (line, i)] = train_data['L%s_S%s_NUM_MAX' % (line, i)] - train_data['L%s_S%s_NUM_MIN' % (line, i)]

        logger.info('line num sec %s end' % i)
        logger.info('size %s %s' % train_data.shape)

    train_data['L_all_hash_cat'] = train_data[LIST_COLUMN_CAT].apply(lambda x: hash(''.join(map(str, x))), axis=1)
    train_data['L_all_hash_num'] = train_data[LIST_COLUMN_NUM].apply(lambda x: hash(''.join(map(str, x))), axis=1)
    train_data['L_all_hash_dat'] = train_data[LIST_COLUMN_DATE].apply(lambda x: hash(''.join(map(str, x))), axis=1)
    train_data['L_all_hash'] = train_data[LIST_FEATURE_COLUMN_NAME].apply(lambda x: hash(''.join(map(str, x))), axis=1)


    for i in list(range(4)) + ['']:
        tmp_cols = [col for col in LIST_COLUMN_CAT if 'L%s' % i in col]
        train_data['L%s_hash_cat'%i] = train_data[tmp_cols].apply(lambda x: hash(''.join(map(str, x))), axis=1)
        tmp_cols = [col for col in LIST_COLUMN_NUM if 'L%s' % i in col]
        train_data['L%s_hash_num'%i] = train_data[tmp_cols].apply(lambda x: hash(''.join(map(str, x))), axis=1)
        tmp_cols = [col for col in LIST_COLUMN_DATE if 'L%s' % i in col]
        train_data['L%s_hash_dat'%i] = train_data[tmp_cols].apply(lambda x: hash(''.join(map(str, x))), axis=1)
        tmp_cols = [col for col in LIST_FEATURE_COLUMN_NAME if 'L%s' % i in col]
        train_data['L%s_hash'%i] = train_data[tmp_cols].apply(lambda x: hash(''.join(map(str, x))), axis=1)

    df = train_data[['Id', TARGET_COLUMN_NAME] +
                    feature_column]

    logger.info('size %s %s' % df.shape)
    df.to_csv('../data/train_etl/train_elt_%s.csv.gz' % num, index=False, compression='gzip')

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
        feature_column += [col+'_%s_D'%i for col in cols]
        feature_column += col_names
        logger.info('line date %s end' % i)

    for i in list(range(4)) + ['']:
        feature_column += ['L%s_NUM_MAX' % i, 'L%s_NUM_MIN' % i, 'L%s_NUM_AVG' % i, 'L%s_NUM_DIFF' % i]

    num_column = [col for col in LIST_FEATURE_COLUMN_NAME if col in LIST_COLUMN_NUM]
    for i in range(52):
        cols = [col for col in num_column if 'S%s' % i in col]
        if len(cols) == 0:
            continue
        line = cols[0][1]
        col_names = ['L%s_S%s_NUM_MAX' % (line, i), 'L%s_S%s_NUM_MIN' % (line, i), 'L%s_S%s_NUM_AVG' % (line, i), 'L%s_S%s_NUM_DIFF' % (line, i)]
        feature_column += col_names

    #feature_column = [col for col in feature_column
    #                  if col not in LIST_COLUMN_ZERO]
    feature_column.append('L_all_hash')
    feature_column.append('L_all_hash_cat')
    feature_column.append('L_all_hash_num')
    feature_column.append('L_all_hash_dat')
    for i in list(range(4)) + ['']:
        feature_column.append('L%s_hash_cat'%i)
        feature_column.append('L%s_hash_num'%i)
        feature_column.append('L%s_hash_dat'%i)
        feature_column.append('L%s_hash'%i)

    feature_column += ['L_hash_cnt', 'L_hash_cnt_cat', 'L_hash_cnt_num', 'L_hash_cnt_date']
    mst = pandas.read_csv('../data/hash_table.csv', header=None, names=['L_hash_cnt'], index_col=0)
    mst_cat = pandas.read_csv('../data/hash_table_cat.csv', header=None, names=['L_hash_cnt_cat'], index_col=0)
    mst_num = pandas.read_csv('../data/hash_table_num.csv', header=None, names=['L_hash_cnt_num'], index_col=0)
    mst_date = pandas.read_csv('../data/hash_table_date.csv', header=None, names=['L_hash_cnt_date'], index_col=0)
    
    feature_column += ['L%s_hash_cnt'%i for i in range(4)]
    list_mst = [{'mst': pandas.read_csv('../data/hash_table_L%s.csv'%i, header=None, names=['L%s_hash_cnt'%i], index_col=0),
                 'cat': pandas.read_csv('../data/hash_table_cat_L%s.csv'%i, header=None, names=['L%s_hash_cnt_cat'%i], index_col=0),
                 'num': pandas.read_csv('../data/hash_table_num_L%s.csv'%i, header=None, names=['L%s_hash_cnt_num'%i], index_col=0),
                 'date': pandas.read_csv('../data/hash_table_date_L%s.csv'%i, header=None, names=['L%s_hash_cnt_date'%i], index_col=0),
                }
                for i in range(4)]


    feature_column += ['L%s_hash_cnt_nat'%i for i in range(4)]
    feature_column += ['L%s_hash_cnt_num'%i for i in range(4)]
    feature_column += ['L%s_hash_cnt_date'%i for i in range(4)]


    path = sys.argv[1]
    train_data_all = pandas.read_csv(path, chunksize=1000)
    num = 0
    file_num = re.match(u'.*_(\d+).csv.gz$', path).group(1)
    for train_data in train_data_all:
        aaa = train_data[LIST_FEATURE_COLUMN_NAME].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
        train_data['__hash__'] = aaa
        train_data = pandas.merge(train_data, mst, how='left', left_on='__hash__', right_index=True,  copy=False)
        aaa = train_data[LIST_COLUMN_NUM].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
        train_data['__hash1__'] = aaa
        train_data = pandas.merge(train_data, mst_num, how='left', left_on='__hash1__', right_index=True,  copy=False)
        aaa = train_data[LIST_COLUMN_CAT].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
        train_data['__hash2__'] = aaa
        train_data = pandas.merge(train_data, mst_cat, how='left', left_on='__hash2__', right_index=True,  copy=False)
        aaa = train_data[LIST_COLUMN_DATE].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
        train_data['__hash3__'] = aaa
        train_data = pandas.merge(train_data, mst_date, how='left', left_on='__hash3__', right_index=True,  copy=False)

        for i in range(4):
            cols = [col for col in LIST_FEATURE_COLUMN_NAME if 'L%s' % i in col]
            aaa = train_data[cols].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
            train_data['__hash_%s__'%i] = aaa
            train_data = pandas.merge(train_data, list_mst[i]['mst'], how='left', left_on='__hash_%s__'%i, right_index=True,  copy=False)

            cols2 = [col for col in cols if col in LIST_COLUMN_NUM]
            aaa = train_data[cols2].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
            train_data['__hash1_%s__'%i] = aaa
            train_data = pandas.merge(train_data, list_mst[i]['num'], how='left', left_on='__hash1_%s__'%i, right_index=True,  copy=False)

            cols2 = [col for col in cols if col in LIST_COLUMN_CAT]
            aaa = train_data[cols].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
            train_data['__hash2_%s__'%i] = aaa
            train_data = pandas.merge(train_data, list_mst[i]['cat'], how='left', left_on='__hash2_%s__'%i, right_index=True,  copy=False)

            cols2 = [col for col in cols if col in LIST_COLUMN_DATE]
            aaa = train_data[cols2].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
            train_data['__hash3_%s__'%i] = aaa
            train_data = pandas.merge(train_data, list_mst[i]['date'], how='left', left_on='__hash3_%s__'%i, right_index=True,  copy=False)


        postfix = '%s_%s' % (file_num, num)
        etl(train_data, postfix, feature_column, date_cols)
        num += 1
        del train_data
        gc.collect()
