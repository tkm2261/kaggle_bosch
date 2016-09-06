# encoding: utf-8

import os
import pandas
import numpy
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data/')

CAT_TRAIN_DATA = os.path.join(DATA_DIR, 'train_categorical.csv.gz')
CAT_TEST_DATA = os.path.join(DATA_DIR, 'test_categorical.csv')

NUM_TRAIN_DATA = os.path.join(DATA_DIR, 'train_numeric.csv')


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

        logger.info('start make colname')
        for i, ptr in enumerate(self.enc.feature_indices_[:-1]):
            col_name = pd_data.columns[i]
            for j, _ in enumerate(range(ptr, self.enc.feature_indices_[i + 1])):
                col_new_name = '%s-T%s' % (col_name, j)
                list_columns.append(col_new_name)
        list_columns = numpy.array(list_columns)
        logger.info('end make colname')

        logger.info('start chi2 test')
        chi_socre, p_val = chi2(enc_data, target)
        logger.info('end chi2 test')
        selected_idx = numpy.argsort(chi_socre)[::-1][:self.n_feature]

        logger.info('p val: %s ~ %s' % (p_val[selected_idx][0], p_val[selected_idx][-1]))
        enc_data = enc_data[:, selected_idx].todense()

        self.selected_idx = selected_idx
        self.chi_socre = chi_socre
        self.list_columns = list_columns

        return pandas.DataFrame(enc_data, columns=list_columns[selected_idx])

    def transform(self, pd_data):
        enc_data = self.enc.transform(pd_data)
        enc_data = enc_data[:, self.selected_idx].todense()

        return pandas.DataFrame(enc_data, columns=self.list_columns[self.selected_idx])


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

if __name__ == '__main__':
    test()
    logger.info('load data 0')
    target = pandas.read_csv(NUM_TRAIN_DATA, usecols=['Response'])['Response']
    logger.info('load data 1')
    pd_data = pandas.read_csv(CAT_TRAIN_DATA, compression='gzip', chunksize=100000)
    logger.info('load data 2')
    pd_data = pd_data.apply(lambda x: int(x[1:]))
    logger.info('load data 3')
    pd_data = pd_data.fillna(0)

    id_col = pd_data['Id']
    pd_data = pd_data.ix[:, 1:]
    chi = ChiExtractor()

    pd_ext_data = chi.fit_transform(pd_data, target)
    pd_ext_data['Id'] = id_col

    
    logger.info('load data 1')
    pd_data = pandas.read_csv(CAT_TEST_DATA)
    logger.info('load data 2')
    pd_data = pd_data.apply(lambda x: int(x[1:]))
    logger.info('load data 3')
    pd_data = pd_data.fillna(0)

    id_col = pd_data['Id']
    pd_data = pd_data.ix[:, 1:]
    chi = ChiExtractor()

    pd_ext_data = chi.fit_transform(pd_data, target)
    pd_ext_data['Id'] = id_col

    
