# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import gc
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TRAIN_POSITIVE_DATA = os.path.join(DATA_DIR, 'train_simple_join_pos.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME
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
        if score > max_score:
            max_score = score
    return max_score


def mcc_scoring2(y_pred_prb, y):
    list_thresh = numpy.arange(1, 100) / 100
    max_score = -1
    idx = None
    for thresh in list_thresh:
        y_pred = numpy.where(y_pred_prb >= thresh, 1, 0)
        score = mcc(y, y_pred)
        if score > max_score:
            max_score = score
            idx = thresh
    return idx, max_score


def min_date(row):
    try:
        return row - min(ele for ele in row if ele > 0)
    except ValueError:
        return row


def predict(_df, list_model, _feature_column, date_cols):
    df = pandas.DataFrame(_df)
    df[date_cols] = df[date_cols].apply(min_date, axis=1)

    feature_column = list(_feature_column)
    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        df['part_L%s' % i] = df[cols].apply(lambda row: 1 if max(row) < 0 else 0, axis=1)
        feature_column.append('part_L%s' % i)

    data = df[feature_column]
    pred = []

    cnt = 0
    for j, jj in enumerate(list(range(4)) + ['']):
        cols = [col for col in feature_column if 'L%s' % jj in col]
        model = list_model[cnt]
        pred.append(model.predict_proba(data[cols])[:, 1])
        cnt += 1
        model = list_model[cnt]
        pred.append(model.predict_proba(data[cols].values)[:, 1])
        cnt += 1

    pred = numpy.array(pred).T
    predict_proba = list_model[-1].predict_proba(pred)[:, 1]

    return predict_proba


def predict2(_df, list_model, _feature_column, date_cols):
    df = pandas.DataFrame(_df)
    df[date_cols] = df[date_cols].apply(min_date, axis=1)

    feature_column = list(_feature_column)
    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        df['part_L%s' % i] = df[cols].apply(lambda row: 1 if max(row) < 0 else 0, axis=1)
        feature_column.append('part_L%s' % i)

    data = df[feature_column]
    cols = [col for col in feature_column if 'L' in col]
    model = list_model[-3]
    predict_proba = model.predict_proba(data[cols])[:, 1]
    return predict_proba

if __name__ == '__main__':
    logger.info('load start')
    with open('list_xgb_model.pkl', 'rb') as f:
        list_model = pickle.load(f)

    train_pos_data = pandas.read_csv(TRAIN_POSITIVE_DATA)
    train_pos_data = train_pos_data.fillna(-1)
    list_train_data = pandas.read_csv(TRAIN_DATA, chunksize=100000)

    logger.info('load end')
    logger.info('pos_data shape %s %s' % train_pos_data.shape)

    pos_target = train_pos_data[TARGET_COLUMN_NAME]
    pos_data = train_pos_data[LIST_FEATURE_COLUMN_NAME]

    pos_data = pandas.read_csv('pos_data_%s.csv.gz' % (170), index_col=0).fillna(-1)
    pos_target = pandas.read_csv('pos_target_%s.csv.gz' % (170), header=None, index_col=0)[1]
    original_col = pos_data.columns.values

    feature_column = [col for col in LIST_FEATURE_COLUMN_NAME if col not in LIST_DUPLICATE_COL_NAME]
    date_cols = [col for col in feature_column if 'D' in col]

    for i, train_data in enumerate(list_train_data):
        logger.info('iter: %s ' % i)
        train_data = train_data.fillna(-1)

        target = train_data[TARGET_COLUMN_NAME]
        data = train_data[LIST_FEATURE_COLUMN_NAME]

        pred = predict2(data, list_model, feature_column, date_cols)  # model.predict_proba(neg_data)[:, 1]
        thresh, score = mcc_scoring2(pred, target.values)
        idx = target.values != pred
        pos_target = pos_target.append(target.ix[idx])
        pos_data = pos_data.append(data.ix[idx, original_col])

        logger.info('%s: pos shape %s' % (i, pos_data.shape[0]))
        pred = predict2(pos_data, list_model, feature_column, date_cols)
        score = roc_auc_score(pos_target, pred)
        thresh, mcc_score = mcc_scoring2(pred, target.values)
        logger.info('INSAMPLE score: %s, thresh: %s, mcc_score: %s' % (score, thresh, mcc_score))

        if (i + 1) % 50 == 0:
            pos_data.to_csv('pos_data_170_%s.csv' % (i + 1))
            pos_target.to_csv('pos_target_170_%s.csv' % (i + 1))
        del train_data
        gc.collect()
