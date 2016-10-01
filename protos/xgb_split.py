# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob
import hashlib

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
from feature_0928 import LIST_COLUMN_ZERO

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)


def hash_groupby(df, feature_column):

    new_feature_column = list(feature_column)
    for col in feature_column:
        if 'hash' not in col:
            continue
        logger.info('hash col%s' % col)
        tmp = df.groupby(col)[[TARGET_COLUMN_NAME]].count()
        #tmp_mean = tmp[TARGET_COLUMN_NAME].mean()
        #tmp[TARGET_COLUMN_NAME][tmp[TARGET_COLUMN_NAME] < 2] = 2
        tmp.columns = [col + '_prob']
        new_feature_column.append(col + '_prob')
        df = pandas.merge(df, tmp, how='left', left_on=col, right_index=True)

    df[[col for col in new_feature_column if 'hash' in col]].to_csv('hash_prob.csv', index=True)
    return df, new_feature_column


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    df = pandas.read_csv(filename)
    return df


if __name__ == '__main__':
    logger.info('load start')
    # train_data = pandas.read_csv(TRAIN_DATA)

    # train_data = pandas.concat(pandas.read_csv(path) for path in glob.glob(
    #    os.path.join(DATA_DIR, 'train_etl/*'))).reset_index(drop=True)
    p = Pool()
    train_data = pandas.concat(p.map(read_csv,
                                     glob.glob(os.path.join(DATA_DIR, 'train_etl/*'))
                                     )).reset_index(drop=True)
    p.close()
    p.join()
    logger.info('shape %s %s' % train_data.shape)
    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']
    feature_column = [col for col in feature_column if col not in LIST_COLUMN_ZERO]
    #feature_column = [col for col in feature_column if 'hash' not in col or 'L_hash_num' == col]

    train_data = train_data[['Id', TARGET_COLUMN_NAME] + feature_column]
    #train_data, feature_column = hash_groupby(train_data, feature_column)

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
    # 2016-09-27/22:03:59 __main__ 141 [INFO][<module>] param: {'learning_rate': 0.1, 'scale_pos_weight': 10, 'max_depth': 10,
    #                                     'min_child_weight': 1, 'colsample_bytree': 0.3, 'n_estimators': 300, 'subsample': 1}
    # 2016-09-27/22:11:58 __main__ 179 [INFO][<module>] score: 0.719638635551
    # 2016-09-27/22:19:54 __main__ 179 [INFO][<module>] score: 0.733182884206
    # 2016-09-27/22:27:51 __main__ 179 [INFO][<module>] score: 0.722172677022
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
                model.fit(data.ix[train_idx, cols], target[train_idx],
                          eval_metric=evalmcc_xgb_min,
                          verbose=False)
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
        model.fit(data[cols], target,
                  eval_metric=evalmcc_xgb_min,
                  verbose=False)

        list_estimator[idx] = model
        idx += 1

    with open('list_xgb_model.pkl', 'wb') as f:
        pickle.dump(list_estimator, f, -1)
