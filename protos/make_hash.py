# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob
import hashlib
import gc
from multiprocessing import Pool
from feature_orig import LIST_COLUMN_CAT, LIST_COLUMN_NUM, LIST_COLUMN_DATE

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TARGET_COLUMN_NAME = u'Response'

from feature import LIST_FEATURE_COLUMN_NAME

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    logger.info(filename)
    ret = None
    for df in pandas.read_csv(filename, chunksize=10000):
        df_ret = pandas.DataFrame()
        df = df[LIST_FEATURE_COLUMN_NAME]
        df_ret['hash_all'] = df.apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
        df_ret['hash_cat'] = df[LIST_COLUMN_CAT].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
        df_ret['hash_num'] = df[LIST_COLUMN_NUM].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
        df_ret['hash_date'] = df[LIST_COLUMN_DATE].apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)

        if ret is None:
            ret = df_ret
        else:
            ret = pandas.concat([ret, df_ret])
        del df
        gc.collect()
        logger.info('%s'%(ret.shape[0]))
    return ret

if __name__ == '__main__':
    list_path = list(glob.glob(os.path.join(DATA_DIR, 'train_simple_part/*')))
    list_path += list(glob.glob(os.path.join(DATA_DIR, 'test_simple_part/*')))
    list_path = sorted(list_path)
    p = Pool(40)
    data = pandas.concat(p.map(read_csv, list_path)).reset_index(drop=True)

    _data = pandas.DataFrame(data['hash_all'].values, columns=['hash']).groupby('hash')['hash'].count()
    _data.to_csv('../data/hash_table.csv')

    _data = pandas.DataFrame(data['hash_cat'].values, columns=['hash']).groupby('hash')['hash'].count()
    _data.to_csv('../data/hash_table_cat.csv')

    _data = pandas.DataFrame(data['hash_num'].values, columns=['hash']).groupby('hash')['hash'].count()
    _data.to_csv('../data/hash_table_num.csv')

    _data = pandas.DataFrame(data['hash_date'].values, columns=['hash']).groupby('hash')['hash'].count()
    _data.to_csv('../data/hash_table_date.csv')
