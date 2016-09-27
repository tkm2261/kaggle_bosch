# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob
from multiprocessing import Pool
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from utils import mcc_optimize, evalmcc_xgb_min
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM

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

from numba.decorators import jit


@jit
def mcc_scoring2(y_pred_prb, y):
    list_thresh = numpy.arange(1, 100) / 100
    max_score = -1
    idx = None
    for thresh in list_thresh:
        y_pred = numpy.where(y_pred_prb >= thresh, 1, 0)
        score = mcc(y, y_pred)
        #logger.debug('thresh: %s, score: %s' % (thresh, score))
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

def read_csv(filename):
    'converts a filename to a pandas dataframe'
    return pandas.read_csv(filename)

if __name__ == '__main__':
    logger.info('load start')
    # train_data = pandas.read_csv(TRAIN_DATA)

    #train_data = pandas.concat(pandas.read_csv(path) for path in glob.glob(
    #    os.path.join(DATA_DIR, 'train_etl/*'))).reset_index(drop=True)
    p = Pool()
    train_data = pandas.concat(p.map(read_csv, 
                                     glob.glob(os.path.join(DATA_DIR, 'train_etl/*'))
                                 )).reset_index(drop=True)
    p.close()
    p.join()
    logger.info('shape %s %s' % train_data.shape)
    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']
    target = train_data[TARGET_COLUMN_NAME].values.astype(numpy.bool_)
    data = train_data[feature_column].fillna(-10)

    pos_rate = float(sum(target)) / target.shape[0]
    logger.info('shape %s %s' % data.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), pos_rate))

    _params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.3,
               'max_depth': 10, 'min_child_weight': 0.01, 'n_estimators': 300,
               'scale_pos_weight': 1.}
    _params = {'subsample': 1, 'scale_pos_weight': 10, 'n_estimators': 100, 'max_depth': 10,
               'colsample_bytree': 0.3, 'min_child_weight': 1, 'learning_rate': 0.1}
    all_params = {'max_depth': [10],
                  'n_estimators': [300],
                  'learning_rate': [0.1],
                  'scale_pos_weight': [10],
                  'min_child_weight': [0.01],
                  'subsample': [1],
                  'colsample_bytree': [0.3],
                  }

    cv = StratifiedKFold(target, n_folds=3, shuffle=True, random_state=0)
    all_ans = None
    all_target = None

    with open('train_feature_1.py', 'w') as f:
        f.write("LIST_TRAIN_COL = ['" + "', '".join(feature_column) + "']\n\n")
    logger.info('cv_start')
    for params in ParameterGrid(all_params):
        logger.info('param: %s' % (params))
        for train_idx, test_idx in list(cv):
            list_estimator = []
            ans = []
            insample_ans = []
            for i in [0, 1, 2, 3, '']:  # [1, 3, '']:  #
                logger.info('model: %s' % i)
                cols = [col for col in feature_column if 'L%s' % i in col]
                model = XGBClassifier(seed=0)
                model.set_params(**params)
                model.fit(data.ix[train_idx, cols], target[train_idx], eval_metric=evalmcc_xgb_min)
                list_estimator.append(model)
                ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
                insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])
            logger.info('train_end')
            ans = numpy.array(ans).T
            insample_ans = numpy.array(insample_ans).T
            if all_ans is None:
                all_ans = ans
                all_target = target[test_idx]
            else:
                all_ans = numpy.r_[all_ans, ans]
                all_target = numpy.r_[all_target, target[test_idx]]

            # model = LogisticRegressionCV(n_jobs=-1, class_weight='balanced', scoring='roc_auc', random_state=0)
            # model = LogisticRegression(n_jobs=-1, class_weight='balanced')
            model = XGBClassifier(seed=0)
            # model.set_params(**params)

            # model.fit(numpy.r_[ans, insample_ans], numpy.r_[target[test_idx], target[train_idx]])
            model.fit(ans, target[test_idx])
            pred = model.predict_proba(ans)[:, 1]  # ans.max(axis=1)
            logger.info('pred thresh: %s, score: %s' % mcc_optimize(pred, target[test_idx]))
            score = roc_auc_score(target[test_idx], ans.mean(axis=1))
            logger.info('mean thresh: %s, score: %s' % mcc_optimize(ans.mean(axis=1), target[test_idx]))
            logger.info('all thresh: %s, score: %s' % mcc_optimize(ans[:, -1], target[test_idx]))
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
    for i in [0, 1, 2, 3, '']:  # [1, 3, '']:
        logger.info('model: %s' % i)
        cols = [col for col in feature_column if 'L%s' % i in col]
        model = XGBClassifier(seed=0)
        model.set_params(**params)
        model.fit(data[cols], target, eval_metric=evalmcc_xgb_min)
        list_estimator[idx] = model
        idx += 1

    with open('list_xgb_model.pkl', 'wb') as f:
        pickle.dump(list_estimator, f, -1)
