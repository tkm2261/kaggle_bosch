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
from sklearn.cross_validation import StratifiedKFold

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

    return cols1, cols2, score

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
        p = Pool()
        list_pross = []
        for j in range(i+1, len(pairs)):
            cols1 = pairs[i]
            cols2 = pairs[j]
            merge_cols = cols1 + cols2 + [TARGET_COLUMN_NAME]
            list_pross.append(p.apply_async(calc_t_score, (df[merge_cols], cols1, cols2)))
        list_pross = [proc.get() for proc in list_pross]
        p.close()
        p.join()
        list_pross = sorted(list_pross, key=lambda x:x[2], reverse=True)
        col1, col2, score = list_pross[0]
        if score >= max_t_score:
            max_t_score = score
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
    target = data[TARGET_COLUMN_NAME].fillna(2)
    cv = StratifiedKFold(target, n_folds=4, shuffle=True, random_state=0)
    _, idx = list(cv)[0]
    data2 = data.ix[idx].copy()
    del data
    p.close()
    p.join()
    main(data2)

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
