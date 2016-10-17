# encoding: utf-8

import os
import pandas
import numpy
import logging
import gc
import glob

from multiprocessing import Pool
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data/')

CAT_TRAIN_DATA = os.path.join(DATA_DIR, 'train_categorical2.csv.gz')
CAT_TEST_DATA = os.path.join(DATA_DIR, 'test_categorical2.csv.gz')

NUM_TRAIN_DATA = os.path.join(DATA_DIR, 'train_numeric.csv.gz')

from feature_orig import LIST_COLUMN_CAT, LIST_COLUMN_NUM
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE, LIST_SAME_COL

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')

logger = logging.getLogger(__name__)


class ChiExtractor:

    def __init__(self, n_feature=200):
        self.enc = OneHotEncoder(dtype=numpy.int)
        self.n_feature = n_feature

        self.list_columns = None
        self.selected_idx = None
        self.chi_socre = None

    def fit_transform(self, pd_data, target):

        logger.info('start encoding')
        self.enc.fit(pd_data)
        enc_data = self.enc.transform(pd_data)

        logger.info('end encoding')
        list_columns = []

        logger.info('start chi2 test')
        chi_socre, p_val = chi2(enc_data, target)
        logger.info('end chi2 test')
        selected_idx = numpy.argsort(chi_socre)[::-1][:self.n_feature]

        logger.info('p val: %s ~ %s' % (p_val[selected_idx][0], p_val[selected_idx][-1]))
        enc_data = enc_data[:, selected_idx].todense()

        logger.info('start make colname')
        """
        for i, ptr in enumerate(self.enc.feature_indices_[:-1]):
            col_name = pd_data.columns[i]
            for j, _ in enumerate(range(ptr, self.enc.feature_indices_[i + 1])):
                col_new_name = '%s-T%s' % (col_name, j)
                list_columns.append(col_new_name)
        list_columns = numpy.array(list_columns)
        """
        logger.info('end make colname')

        self.selected_idx = selected_idx
        self.chi_socre = chi_socre
        self.list_columns = list_columns

        return pandas.DataFrame(enc_data, columns=selected_idx)

    def transform(self, pd_data):
        enc_data = self.enc.transform(pd_data)
        enc_data = enc_data[:, self.selected_idx].todense()

        return pandas.DataFrame(enc_data, columns=self.selected_idx)


def test():
    data = [[0, 0, 3],
            [1, 1, 0],
            [0, 2, 1],
            [1, 0, 2]]
    target = [1, 0, 1, 0]
    pd_data = pandas.DataFrame(data, columns=['a', 'b', 'c'])
    chi = ChiExtractor()
    pd_ext_data = chi.fit_transform(pd_data, target)

    print(pd_ext_data)
    """
       a-T1  a-T0  c-T3  c-T2  c-T1  c-T0  b-T2  b-T1  b-T0
    0     0     1     1     0     0     0     0     0     1
    1     1     0     0     0     0     1     0     1     0
    2     0     1     0     0     1     0     1     0     0
    3     1     0     0     1     0     0     0     0     1
    """


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    feature_column = LIST_COLUMN_NUM

    df = pandas.read_csv(filename, usecols=['Id', 'Response'] + feature_column, dtype=str)
    return df


def read_df(df, i):
    df.to_csv(os.path.join(DATA_DIR, 'train_rank/train_rank_%s.csv.gz' % i), index=False, compression='gzip')
    logger.info('load end %s' % i)


if __name__ == '__main__':
    test()
    logger.info('load data 0')
    p = Pool()

    train_data = pandas.concat(p.map(read_csv,
                                     glob.glob(os.path.join(DATA_DIR, 'train_simple_part/*'))
                                     )).reset_index(drop=True)
    p.close()
    p.join()

    def aaa(data_col):
        col = data_col.columns.values[0]
        logger.info(col)

        data_col = data_col.rank(method='min').fillna(0).astype(int)
        #cnt = data_col.groupby(col)[col].count()
        #cnt = (cnt < 4).index.values
        #data_col[data_col[col].isin(cnt)] = cnt.min()
        return data_col

    p = Pool()
    cols = [col for col in train_data.columns.values if col != 'Id' and col != 'Response']
    list_col = p.map(aaa, [train_data[[col]] for col in cols])
    logger.info('p end')
    ids = train_data['Id'].values
    target = train_data['Response'].values
    del train_data
    gc.collect()
    p.close()
    p.join()

    train_data = pandas.concat(list_col, axis=1)

    train_data['Id'] = ids
    train_data['Response'] = target

    num = 10000
    res = []
    p = Pool()
    idx = train_data.index.values
    for i in range(int(len(idx) / num) + 1):
        if i * num > len(idx):
            break

        if (i + 1) * num > len(idx):
            ix = idx[i * num:]
        else:
            ix = idx[i * num: (i + 1) * num]

        df = train_data.ix[ix]
        res.append(p.apply_async(read_df, (df, i)))
        i += 1
    [r.get() for r in res]

    p.close()
    p.join()
    logger.info('load data 1 %s %s' % train_data.shape)

    id_col = train_data['Id']
    pd_data = train_data[[col for col in train_data.columns.values if col != 'Id' and col != 'Response']]
    cols = pd_data.columns.values

    target = train_data['Response']
    chi = ChiExtractor()
    pd_data = pandas.DataFrame(pd_data, columns=cols)

    gc.collect()
    pd_ext_data = chi.fit_transform(pd_data, target)
    pd_ext_data['Id'] = id_col
    pd_ext_data.to_csv('chi_cat_rank.csv', index=False)
