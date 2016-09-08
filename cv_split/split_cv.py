# encoding: utf-8

import os
import pandas
import numpy
import logging
from sklearn.cross_validation import StratifiedKFold

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data/')
NUM_TRAIN_DATA = os.path.join(DATA_DIR, 'train_numeric.csv.gz')

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('load start')
    target = pandas.read_csv(NUM_TRAIN_DATA, usecols=['Id', 'Response'])
    logger.info('load end')

    cv = StratifiedKFold(target['Response'], n_folds=3, shuffle=True, random_state=0)

    for i, (train_idx, test_idx) in enumerate(cv):
        logger.info('writing %s' % i)
        target.ix[test_idx].to_csv('bosch_cv_split_%s.csv' % i, index=False)
