# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob
import hashlib
import gc

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
from omit_pos_id import LIST_OMIT_POS_ID, LIST_OMIT_POS_MIN

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

def make_cross(df, feature_columns):
    mst = pandas.read_csv('cross_term.csv', header=None, index_col=0)[1]
    mst = mst[mst > 500]
    for pair in mst.index.values:
        f1, f2 = pair.split('-')
        df[pair] = df[f1] * df[f2]
        feature_columns.append(pair)
        logger.info('cross: %s'%pair)
    return df, feature_columns

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

    train_data = train_data[['Id', TARGET_COLUMN_NAME] + feature_column]
    train_data, feature_column = make_cross(train_data, feature_column)

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
    #2016-10-01/12:49:11 __main__ 115 [INFO][<module>] param: {'min_child_weight': 0.01, 'scale_pos_weight': 1, 'n_est
    #imators': 200, 'max_depth': 9, 'subsample': 1, 'colsample_bytree': 0.5, 'learning_rate': 0.1}
    #2016-10-01/12:49:11 __main__ 122 [INFO][<module>] model: 2
    #2016-10-01/12:49:45 __main__ 122 [INFO][<module>] model:
    #2016-10-01/12:55:12 __main__ 132 [INFO][<module>] train_end
    #2016-10-01/12:55:17 __main__ 152 [INFO][<module>] pred thresh: 0.235527, score: 0.285598045635
    #2016-10-01/12:55:17 __main__ 154 [INFO][<module>] mean thresh: 0.0739454, score: 0.268554492546
    #2016-10-01/12:55:17 __main__ 155 [INFO][<module>] all thresh: 0.157561, score: 0.281773786143
    #2016-10-01/12:55:17 __main__ 156 [INFO][<module>] score: 0.817277189769

    all_params = {'max_depth': [9],
                  'n_estimators': [200],
                  'learning_rate': [0.1],
                  'scale_pos_weight': [1],
                  'min_child_weight': [0.01],
                  'subsample': [1],
                  'colsample_bytree': [0.5],
                  }

    cv = StratifiedKFold(target, n_folds=3, shuffle=True, random_state=0)
    all_ans = None
    all_target = None
    all_ids = None

    omit_idx = train_data[~train_data['Id'].isin(LIST_OMIT_POS_ID)].index.values
    with open('train_feature_1.py', 'w') as f:
        f.write("LIST_TRAIN_COL = ['" + "', '".join(feature_column) + "']\n\n")
    logger.info('cv_start')
    for params in ParameterGrid(all_params):
        logger.info('param: %s' % (params))
        for train_idx, test_idx in list(cv):
            train_omit_idx = numpy.intersect1d(train_idx, omit_idx)
            logger.info('ommit size: %s %s'%(train_idx.shape[0], len(train_omit_idx)))
            list_estimator = []

            ans = []
            insample_ans = []
            for i in [1, 3, '']:  #
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
                all_ids = train_data.ix[test_idx, 'Id'].values
            else:
                all_ans = numpy.r_[all_ans, ans]
                all_target = numpy.r_[all_target, target[test_idx]]
                all_ids = numpy.r_[all_ids, train_data.ix[test_idx, 'Id']]

            model = XGBClassifier(seed=0)
            model.fit(ans, target[test_idx])
            pred = model.predict_proba(ans)[:, 1]
            logger.info('model thresh: %s, score: %s' % mcc_optimize(pred, target[test_idx]))
            pred = ans.max(axis=1)
            logger.info('max thresh: %s, score: %s' % mcc_optimize(pred, target[test_idx]))
            pred = ans.min(axis=1)
            logger.info('min thresh: %s, score: %s' % mcc_optimize(pred, target[test_idx]))
            score = roc_auc_score(target[test_idx], ans[:, -1])
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
    pandas.DataFrame(all_ids).to_csv('stack_1_id_1.csv', index=False)
    del train_data
    idx = 0
    for i in [1, 3, '']:
        gc.collect()
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
