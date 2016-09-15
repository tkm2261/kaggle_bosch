# encoding: utf-8

import pickle
import os
import pandas
import logging
import numpy
import gc

from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME

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
    with open('list_xgb_model.pkl', 'rb') as f:
        list_model = pickle.load(f)
    feature_column = [col for col in LIST_FEATURE_COLUMN_NAME if col not in LIST_DUPLICATE_COL_NAME]
    all_df = pandas.read_csv(TEST_DATA, compression='gzip', chunksize=100000)
    logger.info('end load')

    df_ans = pandas.DataFrame()
    for i, df in enumerate(all_df):
        df = df.fillna(-1)
        data = df[feature_column]
        pred = []
        cnt = 0
        for j, jj in enumerate(list(range(4)) + ['']):
            cols = [col for col in feature_column if 'L%s' % jj in col]
            model = list_model[cnt]
            pred.append(model.predict_proba(data[cols].values)[:, 1])
            cnt += 1
            model = list_model[cnt]
            pred.append(model.predict_proba(data[cols].values)[:, 1])
            cnt += 1

        pred = numpy.array(pred).T
        predict_proba = list_model[-1].predict_proba(pred)[:, 1]
        logger.info('end load')

        predict = numpy.where(predict_proba > 0.6, 1, 0)
        logger.info('end predict')
        ans = pandas.DataFrame(df['Id'])
        ans['Response'] = predict
        ans['proba'] = predict_proba

        df_ans = df_ans.append(ans)
        logger.info('chunk %s: %s' % (i, df_ans.shape[0]))
        del df
        gc.collect()

    df_ans.to_csv('submit.csv', index=False)

if __name__ == '__main__':
    main()
