# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
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
        logger.info('thresh: %s, score: %s' % (thresh, score))
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
        logger.debug('thresh: %s, score: %s' % (thresh, score))
        if score > max_score:
            max_score = score
            idx = thresh
    return idx, max_score


def min_date(row):
    try:
        return row - min(ele for ele in row if ele > 0)
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

if __name__ == '__main__':
    logger.info('load start')
    #feature_column = [col for col in LIST_FEATURE_COLUMN_NAME if col not in LIST_DUPLICATE_COL_NAME]
    feature_column = [col for col in LIST_FEATURE_COLUMN_NAME
                      if col not in LIST_DUPLIDATE_CAT]
    feature_column = [col for col in feature_column
                      if col not in LIST_DUPLIDATE_DATE]
    feature_column = [col for col in feature_column
                      if col not in LIST_SAME_COL]
    logger.info('start_feature_num: %s %s' % (len(LIST_FEATURE_COLUMN_NAME), len(feature_column)))
    train_data = pandas.read_csv('pos_data_170.csv.gz', index_col=0).reset_index(drop=True)
    # train_data = pandas.read_csv(TRAIN_DATA, usecols=feature_column + [TARGET_COLUMN_NAME])
    train_data = train_data.fillna(-1)
    logger.info('load end')

    date_cols = [col for col in feature_column if 'D' in col]
    d_cols = [col + "_DUR" for col in date_cols]
    train_data[d_cols] = train_data.ix[:, date_cols].apply(min_date, axis=1)

    logger.info('date end')

    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        train_data['part_L%s' % i] = train_data[cols].apply(lambda row: 1 if max(row) < 0 else 0, axis=1)
        feature_column.append('part_L%s' % i)
    logger.info('part end')

    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        col_names = ['L%s_D_MIN' % i, 'L%s_D_AVG' % i, 'L%s_D_MAX' % i, 'L%s_D_DIFF' % i]
        #train_data[col_names[0]] = train_data[cols].min(axis=1)
        #train_data[col_names[1]] = train_data[cols].mean(axis=1)
        #train_data[col_names[2]] = train_data[cols].max(axis=1)
        #train_data[col_names[3]] = train_data[col_names[2]] - train_data[col_names[0]]
        train_data[col_names] = train_data[cols].apply(lambda row: date_stats(row, col_names), axis=1)
        feature_column += col_names
        logger.info('line date %s end' % i)
    logger.info('load end')
    logger.info('shape %s %s' % train_data.shape)

    target = pandas.read_csv('pos_target_170.csv.gz', header=None).ix[:, 1].values
    # target = train_data[TARGET_COLUMN_NAME].values.astype(numpy.bool_)
    data = train_data[feature_column]

    pos_rate = float(sum(target)) / target.shape[0]
    logger.info('shape %s %s' % data.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), pos_rate))

    params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.3,
              'max_depth': 5, 'min_child_weight': 0.01, 'n_estimators': 300,
              'scale_pos_weight': 10}

    cv = StratifiedKFold(target, n_folds=10, shuffle=True, random_state=0)
    all_ans = None
    all_target = None

    with open('train_feature_1.py', 'w') as f:
        f.write("LIST_TRAIN_COL = ['" + "', '".join(feature_column) + "']\n\n")
    logger.info('cv_start')
    for train_idx, test_idx in list(cv):
        list_estimator = []
        ans = []
        insample_ans = []
        for i in [1, 3, '']:
            cols = [col for col in feature_column if 'L%s' % i in col]
            model = XGBClassifier(seed=0)
            model.set_params(**params)
            model.fit(data.ix[train_idx, cols], target[train_idx])
            list_estimator.append(model)
            ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
            insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])

        ans = numpy.array(ans).T
        insample_ans = numpy.array(insample_ans).T
        if all_ans is None:
            all_ans = ans
            all_target = target[test_idx]
        else:
            all_ans = numpy.r_[all_ans, ans]
            all_target = numpy.r_[all_target, target[test_idx]]

        model = LogisticRegressionCV(n_jobs=-1, class_weight='balanced', scoring='roc_auc', random_state=0)
        # model = LogisticRegression(n_jobs=-1, class_weight='balanced')
        # model = XGBClassifier(seed=0)
        # model.set_params(**params)

        # model.fit(numpy.r_[ans, insample_ans], numpy.r_[target[test_idx], target[train_idx]])
        model.fit(ans, target[test_idx])
        pred = model.predict_proba(ans)[:, 1]  # ans.max(axis=1)
        print(mcc_scoring2(pred, target[test_idx]))
        score = roc_auc_score(target[test_idx], ans.mean(axis=1))
        logger.info('score: %s' % score)
        score = roc_auc_score(target[test_idx], pred)
        logger.info('INSAMPLE score: %s' % score)
        pred = model.predict_proba(insample_ans)[:, 1]  # ans.max(axis=1)
        score = roc_auc_score(target[train_idx], pred)
        logger.info('INSAMPLE train score: %s' % score)

        list_estimator.append(model)

    pandas.DataFrame(all_ans).to_csv('stack_1_data_1.csv', index=False)
    pandas.DataFrame(all_target).to_csv('stack_1_target_1.csv', index=False)

    idx = 0
    for i in [1, 3, '']:
        cols = [col for col in feature_column if 'L%s' % i in col]
        model = XGBClassifier(seed=0)
        model.set_params(**params)
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

    with open('list_xgb_model.pkl', 'wb') as f:
        pickle.dump(list_estimator, f, -1)
