# encoding: utf-8

import pickle
import os
import pandas
import logging
import numpy
import gc
import glob
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM

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


def mcc(y_true, y_pred):
    n = y_true.shape[0]
    true_pos = sum(1. for i in range(n) if y_true[i] == 1 and y_pred[i] == 1)
    true_neg = sum(1. for i in range(n) if y_true[i] == 0 and y_pred[i] == 0)
    false_pos = sum(1. for i in range(n) if y_true[i] == 0 and y_pred[i] == 1)
    false_neg = sum(1. for i in range(n) if y_true[i] == 1 and y_pred[i] == 0)
    """
    s = (true_pos + false_neg) / n
    p = (true_pos + false_pos) / n

    a = true_pos / n - s * p
    b = numpy.sqrt(s * p * (1 - s) * (1 - p))
    """
    a = true_pos * true_neg - false_pos * false_neg
    b = (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)
    return a / numpy.sqrt(b)


def mcc_scoring(estimator, X, y):

    y_pred_prb = estimator.predict_proba(X)[:, 1]
    list_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    max_score = -1
    for thresh in list_thresh:
        y_pred = numpy.where(y_pred_prb >= thresh, 1, 0)
        score = mcc(y, y_pred)
        logger.info('thresh: %s, score: %s' % (thresh, score))
        if score > max_score:
            max_score = score
    return max_score


def min_date(row):
    try:
        return row - min(ele for ele in row if ele > 0)
    except ValueError:
        return row


def min_date2(row):
    try:
        return row - numpy.mean(row)
    except ValueError:
        return row


def date_stats(row, col_names):
    row_na = row[row > 0]
    if len(row_na) == 0:
        return pandas.Series([-10, -10, -10, -10], index=col_names)
    else:
        r_min = row_na.min()
        r_mean = row_na.mean()
        r_max = row_na.max()
        return pandas.Series([r_min, r_mean, r_max, r_max - r_min], index=col_names)


def main():
    logger.info('start load')
    with open('list_xgb_model.pkl', 'rb') as f:
        list_model = pickle.load(f)

    with open('stack_model_1.pkl', 'rb') as f:
        fin_model = pickle.load(f)

    from train_feature_1 import LIST_TRAIN_COL
    feature_column = LIST_TRAIN_COL

    df_ans = pandas.DataFrame()
    #all_df = pandas.read_csv(TEST_DATA, compression='gzip', chunksize=10000)
    # for i, df in enumerate(all_df):
    # for i in range(12):
    for i, path in enumerate(glob.glob(os.path.join(DATA_DIR, 'test_etl/*'))):
        #df = pandas.read_hdf('predict.h5', key=str(i))
        df = pandas.read_csv(path)
        data = df[feature_column].fillna(-10)
        pred = []

        cnt = 0
        for j, jj in enumerate([0, 1, 2, 3, '']):
            cols = [col for col in feature_column if 'L%s' % jj in col]
            model = list_model[cnt]
            pred.append(model.predict_proba(data[cols])[:, 1])
            cnt += 1

        pred = numpy.array(pred).T
        predict_proba = fin_model.predict_proba(pred)[:, 1]
        predict_proba2 = pred.mean(axis=1)
        logger.info('end load')

        predict = numpy.where(predict_proba >= 0.68, 1, 0)
        logger.info('end predict')
        ans = pandas.DataFrame(df['Id'])
        ans['Response'] = predict
        ans['proba'] = predict_proba
        ans['proba2'] = predict_proba2
        for m in range(pred.shape[1]):
            ans['m%s' % m] = pred[:, m]
        df_ans = df_ans.append(ans)
        logger.info('chunk %s: %s' % (i, df_ans.shape[0]))
        del df
        gc.collect()

    df_ans.to_csv('submit.csv', index=False)

if __name__ == '__main__':
    main()
