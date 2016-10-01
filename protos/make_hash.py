# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob
import hashlib
from multiprocessing import Pool

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
        df = df[LIST_FEATURE_COLUMN_NAME]
        df = df.apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
        if ret is None:
            ret = df
        else:
            ret = pandas.concat([ret, df])
        logger.info('%s'%(ret.shape[0]))
    return ret

if __name__ == '__main__':
    list_path = list(glob.glob(os.path.join(DATA_DIR, 'train_simple_part/*')))
    list_path += list(glob.glob(os.path.join(DATA_DIR, 'test_simple_part/*')))
    list_path = sorted(list_path)
    p = Pool()    
    data = pandas.concat(p.map(read_csv, list_path)).reset_index(drop=True)
    data = pandas.DataFrame(data, columns=['hash']).groupby('hash')['hash'].count()
    data.to_csv('../data/hash_table.csv')
