# encoding: utf-8

import pickle
import os
import pandas
import logging
import numpy
import gc

from feature import LIST_FEATURE_COLUMN_NAME

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


def main():
    logger.info('start load')
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    all_df = pandas.read_csv(TEST_DATA, compression='gzip', chunksize=100000)
    logger.info('end load')

    df_ans = pandas.DataFrame()
    for i, df in enumerate(all_df):
        df = df.fillna(0)
        data = df[LIST_FEATURE_COLUMN_NAME]
        logger.info('end load')
        predict_proba = model.predict_proba(data)[:, 1]
        predict = numpy.where(predict_proba > 0.9, 1, 0)
        logger.info('end predict')
        ans = pandas.DataFrame(df['Id'])
        ans['Response'] = predict
        ans['proba'] = predict_proba

        df_ans = df_ans.append(ans)
        logger.info('chunk %s: %s'%(i, df_ans.shape[0]))
        del df
        gc.collect()
        
    df_ans.to_csv('submit.csv', index=False)

if __name__ == '__main__':
    main()
