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

from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME

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


if __name__ == '__main__':
    logger.info('load start')
    train_data = pandas.read_csv(TRAIN_DATA)
    train_data = train_data.fillna(-1)
    target = train_data[TARGET_COLUMN_NAME].values
    logger.info('load end')
    date_cols = [col for col in LIST_FEATURE_COLUMN_NAME if 'D' in col]

    feature_column = [col for col in LIST_FEATURE_COLUMN_NAME if col not in LIST_DUPLICATE_COL_NAME]

    def min_date(row):
        try:
            return row - min(ele for ele in row if ele > 0)
        except ValueError:
            return row
    train_data[date_cols] = train_data[date_cols].apply(min_date, axis=1)
    logger.info('date end')
    for i in range(4):
        cols = [col for col in date_cols if 'L%s' % i in col]
        train_data['part_L%s' % i] = train_data[cols].apply(lambda row: 1 if max(row) < 0 else 0, axis=1)

        feature_column.append('part_L%s' % i)

    logger.info('load end')
    logger.info('shape %s %s' % train_data.shape)

    data = train_data[feature_column]

    logger.info('shape %s %s' % data.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), float(sum(target)) / target.shape[0]))

    params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.3,
              'max_depth': 5, 'min_child_weight': 0.01, 'n_estimators': 300,
              'scale_pos_weight': 10}
    list_estimator = []
    cv = StratifiedKFold(target, n_folds=10, shuffle=True, random_state=0)
    all_ans = None
    all_target = None

    for train_idx, test_idx in cv:
        ans = []
        insample_ans = []
        for i in list(range(4)) + ['']:
            cols = [col for col in feature_column if 'L%s' % i in col]
            model = XGBClassifier(seed=0)
            model.set_params(**params)
            model.fit(data[cols].values[train_idx], target[train_idx])
            list_estimator.append(model)
            ans.append(model.predict_proba(data[cols].values[test_idx])[:, 1])
            insample_ans.append(model.predict_proba(data[cols].values[train_idx])[:, 1])

            model = LogisticRegressionCV(n_jobs=-1, class_weight='balanced', scoring='roc_auc')
            model.fit(data[cols].values[train_idx], target[train_idx])
            list_estimator.append(model)
            ans.append(model.predict_proba(data[cols].values[test_idx])[:, 1])
            insample_ans.append(model.predict_proba(data[cols].values[train_idx])[:, 1])

        ans = numpy.array(ans).T
        insample_ans = numpy.array(insample_ans).T
        if all_ans is None:
            all_ans = ans
            all_target = target[test_idx]
        else:
            all_ans = numpy.r_[all_ans, ans]
            all_target = numpy.r_[all_target, target[test_idx]]

        model = LogisticRegressionCV(n_jobs=-1, class_weight='balanced', scoring='roc_auc', random_state=0)
        #model = LogisticRegression(n_jobs=-1, class_weight='balanced')
        #model = XGBClassifier(seed=0)
        # model.set_params(**params)

        #model.fit(numpy.r_[ans, insample_ans], numpy.r_[target[test_idx], target[train_idx]])
        model.fit(ans, target[test_idx])
        pred = model.predict_proba(ans)[:, 1]  # ans.max(axis=1)
        print(mcc_scoring2(pred, target[test_idx]))
        score = roc_auc_score(target[test_idx], pred)
        logger.info('INSAMPLE score: %s' % score)
        pred = model.predict_proba(insample_ans)[:, 1]  # ans.max(axis=1)
        score = roc_auc_score(target[train_idx], pred)
        logger.info('INSAMPLE train score: %s' % score)

        list_estimator.append(model)

    pandas.DataFrame(all_ans).to_csv('stack_1_data_1.csv', index=False)
    pandas.DataFrame(all_target).to_csv('stack_1_target_1.csv', index=False)

    idx = 0
    for i in list(range(4)) + ['']:
        cols = [col for col in feature_column if 'L%s' % i in col]
        model = XGBClassifier(seed=0)
        model.set_params(**params)
        model.fit(data[cols].values, target)
        list_estimator[idx] = model
        idx += 1

        model = LogisticRegressionCV(n_jobs=-1, class_weight='balanced', scoring='roc_auc', random_state=0)
        model.fit(data[cols].values, target)
        list_estimator[idx] = model
        idx += 1

    """
    params = {'max_depth': [3, 5, 10],
              'learning_rate': [0.01, 0.1, 1],
              'min_child_weight': [0.01, 0.1, 1],
              'subsample': [0.1, 0.5, 1],
              'colsample_bytree': [0.3, 0.5, 1],
    }

    params = {'max_depth': [3],
              'learning_rate': [0.1],
              'min_child_weight': [0.01],
              'subsample': [1],
              'colsample_bytree': [0.3],
              'n_estimators': [200],
              }
    
    cv = GridSearchCV(model,
                      params,
                      scoring='roc_auc',
                      n_jobs=1,
                      refit=False,
                      verbose=10
                      #scoring=mcc_scoring,
    )
    cv.fit(data, target)
    logger.info('best param: %s'%cv.best_params_)
    logger.info('best score: %s'%cv.best_score_)
    
    model.set_params(**cv.best_params_)
    model.fit(data, target)

    score = roc_auc_score(target, model.predict_proba(data)[:, 1])
    logger.info('INSAMPLE score: %s' % score)
    """
    with open('list_xgb_model.pkl', 'wb') as f:
        pickle.dump(list_estimator, f, -1)
