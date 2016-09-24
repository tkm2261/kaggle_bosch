# encoding: utf-8

import pickle
import os
import pandas
import logging
import numpy
import gc

from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE

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
    _feature_column = LIST_TRAIN_COL

    feature_column = [col for col in LIST_FEATURE_COLUMN_NAME
                      if col not in LIST_DUPLIDATE_CAT]
    feature_column = [col for col in feature_column
                      if col not in LIST_DUPLIDATE_DATE]
    feature_column = [col for col in feature_column
                      if col not in LIST_SAME_COL]
    date_cols = [col for col in feature_column if 'D' in col]
    for i in range(4):
        feature_column += ['part_L%s' % i]
    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        col_names = ['L%s_D_MIN' % i, 'L%s_D_AVG' % i, 'L%s_D_MAX' % i, 'L%s_D_DIFF' % i]
        feature_column += col_names

    logger.info('featuer error: %s' % ([col for col in _feature_column if col not in feature_column]))
    logger.info('end load')
    logger.info('feature_num: %s %s' % (len(LIST_FEATURE_COLUMN_NAME), len(feature_column)))
    df_ans = pandas.DataFrame()
    #all_df = pandas.read_csv(TEST_DATA, compression='gzip', chunksize=10000)

    # for i, df in enumerate(all_df):
    for i in range(12):
        df = pandas.read_hdf('predict.h5', key=str(i))
        df = df.fillna(-1)
        df[[col + "_DUR" for col in date_cols]] = df[date_cols].apply(min_date, axis=1)
        logger.info('date end')
        for i in range(4):
            cols = [col for col in date_cols if 'L%s' % i in col]
            col_names = ['L%s_D_MIN' % i, 'L%s_D_AVG' % i, 'L%s_D_MAX' % i, 'L%s_D_DIFF' % i]
            #df[col_names[0]] = df[cols].min(axis=1)
            #df[col_names[1]] = df[cols].mean(axis=1)
            #df[col_names[2]] = df[cols].max(axis=1)
            #df[col_names[3]] = df[col_names[2]] - df[col_names[0]]
            df[col_names] = df[cols].apply(lambda row: date_stats(row, col_names), axis=1)
        logger.info('date2 end')
        for i in range(4):
            cols = [col for col in date_cols if 'L%s' % i in col]
            df['part_L%s' % i] = df[cols].apply(lambda row: 1 if max(row) < 0 else 0, axis=1)
        logger.info('part end')
        data = df[feature_column]
        pred = []

        cnt = 0
        for j, jj in enumerate([1, 3, '']):
            cols = [col for col in feature_column if 'L%s' % jj in col]
            model = list_model[cnt]
            pred.append(model.predict_proba(data[cols])[:, 1])
            cnt += 1

        pred = numpy.array(pred).T
        predict_proba = fin_model.predict_proba(pred)[:, 1]
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
