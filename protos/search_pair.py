# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob
import hashlib
import gc
from scipy import stats
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


def calc_t_score(df, cols1, cols2):
    merge_cols = cols1 + cols2
    
    df_hash = df[merge_cols].astype(str).sum(axis=1)#.apply(lambda row: hashlib.sha1((','.join(map(str, row))).encode('utf-8')).hexdigest(), axis=1)
    df_hash = pandas.DataFrame(df_hash, columns=['hash'])
    df_hash[TARGET_COLUMN_NAME] = df[TARGET_COLUMN_NAME]
    df_cnt = df_hash.groupby('hash')[['hash']].count()
    df_cnt.columns = ['cnt']

    df_hash = df_hash.merge(df_cnt, how='left', left_on='hash', right_index=True)
    df_hash = df_hash[df_hash[TARGET_COLUMN_NAME] == df_hash[TARGET_COLUMN_NAME]]

    df_hash_pos = df_hash[df_hash[TARGET_COLUMN_NAME] == 1]
    df_hash_neg = df_hash[df_hash[TARGET_COLUMN_NAME] == 0]

    score= df_hash_pos['cnt'].mean() - df_hash_neg['cnt'].mean()

    score = numpy.fabs(score)

    return score

def main(df):
    pairs = [[col] for col in df.columns.values if col != TARGET_COLUMN_NAME and col != 'Id']
    with open('pair.txt', 'w') as f:
        f.write('')
    while 1:
        logger.info(pairs.__repr__())
        with open('pair.txt', 'a') as f:
            f.write(pairs.__repr__())
            f.write('\n')
        pairs = search(df, pairs)
        logger.info('pair_num: %s'%len(pairs))
        if len(pairs) == 1:
            break
map_cache = {}

def search(df, pairs):
    max_t_score = 0
    merge_cand = None
    for i in range(len(pairs) - 1):
        logger.info('progress: %s/%s'%(i, len(pairs)))
        for j in range(i+1, len(pairs)):
            logger.info('    progress: %s'%(j))
            cols1 = pairs[i]
            cols2 = pairs[j]

            str1 = ''.join(cols1)
            str2 = ''.join(cols2)
            if (str1, str2) in map_cache:
                t_score = map_cache[str1, str2]
            else:
                t_score = calc_t_score(df, cols1, cols2)
                map_cache[str1, str2] = t_score
            if t_score >= max_t_score:
                t_score = max_t_score 
                merge_cand = [cols1, cols2]

    new_pairs = [p for p in pairs if p not in merge_cand]
    ret = []
    for pair in merge_cand:
        ret.extend(pair)
    new_pairs.append(ret)
    return new_pairs

def read_csv(filename):
    'converts a filename to a pandas dataframe'
    logger.info(filename)

    return pandas.read_csv(filename)

def test():
    list_path = list(glob.glob(os.path.join(DATA_DIR, 'train_etl/*')))
    list_path += list(glob.glob(os.path.join(DATA_DIR, 'test_etl/*')))
    list_path = sorted(list_path)
    p = Pool()
    data = pandas.concat(p.map(read_csv, list_path)).reset_index(drop=True)
    p.close()
    p.join()
    main(data)

if __name__ == '__main__':
    test()
    """
    list_path = list(glob.glob(os.path.join(DATA_DIR, 'train_simple_part/*')))[:1]
    list_path += list(glob.glob(os.path.join(DATA_DIR, 'test_simple_part/*')))[:1]
    list_path = sorted(list_path)
    p = Pool(40)
    data = pandas.concat(map(read_csv, list_path)).reset_index(drop=True)
    p.close()
    p.join()
    
    calc_t_score(data, ['L3_S43_F4060'], ['L1_S24_F1525'])
    """
