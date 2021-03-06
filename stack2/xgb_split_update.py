# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import glob
import hashlib
import gc
import gzip

from multiprocessing import Pool
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from utils import mcc_optimize, evalmcc_xgb_min, make_cv
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM
from feature_0928 import LIST_COLUMN_ZERO
from feature_1009 import LIST_COLUMN_ZERO_MIX
from omit_pos_id import LIST_OMIT_POS_ID, LIST_OMIT_POS_MIN
from feature_1026 import LIST_ZERO_COL, LIST_ZERO_COL2, LIST_ZERO_COL3
from feature_1026_cnt import LIST_ZERO_COL_CNT, LIST_ZERO_COL_CNT2
from feature_1026_all import LIST_ZERO_COL_ALL

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
        logger.info('cross: %s' % pair)
    return df, feature_columns


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


def make_stack(df, feature_columns):
    logger.info('STACKING!!')
    ids = pandas.read_csv('stack_1_id_1.csv')['0'].values
    data = pandas.read_csv('stack_1_data_1.csv')

    new_cols = ['L_pred_%s' % col for col in data.columns.values]
    data.columns = new_cols
    feature_columns += new_cols

    data['Id'] = ids
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')
    return df, feature_columns


def make_magic(df, feature_columns):
    logger.info('make magic')
    #df['L_Id'] = df['Id'].values
    # feature_column.append('L_Id')

    idx = df.index.values
    df.sort_values('L_D_MIN', inplace=True)

    df['L_MAGIC3'] = df['Id'].diff().values
    feature_column.append('L_MAGIC3')

    df = df.ix[idx]
    logger.info('shape: %s %s' % df.shape)
    return df, feature_columns


def read_df(df):
    return df


def get_feature_importance(model):
    b = model
    fs = b.get_fscore()
    all_features = [fs.get(f, 0.) for f in b.feature_names]
    all_features = numpy.array(all_features, dtype=numpy.float32)
    return all_features / all_features.sum()


if __name__ == '__main__':
    logger.info('load start')
    p = Pool()
    train_data = pandas.concat(p.map(read_csv,
                                     glob.glob(os.path.join(DATA_DIR, 'train_join/*'))
                                     )).reset_index(drop=True)

    p.close()
    p.join()

    logger.info('shape %s %s' % train_data.shape)

    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']
    train_data = train_data[['Id', TARGET_COLUMN_NAME] + feature_column]
    gc.collect()
    # with open('train_data.pkl.gz', 'wb') as f:
    #    pickle.dump(train_data, f, -1)

    feature_column = [col for col in feature_column if col not in LIST_COLUMN_ZERO_MIX +
                      LIST_ZERO_COL + LIST_ZERO_COL2 + LIST_ZERO_COL3 +
                      LIST_ZERO_COL_CNT + LIST_ZERO_COL_CNT2 +
                      LIST_ZERO_COL_ALL]
    feature_column = [col for col in feature_column if 'hash' not in col or 'cnt' in col]

    target = pandas.Series(train_data[TARGET_COLUMN_NAME].values.astype(numpy.bool_), index=train_data['Id'].values)
    data = train_data[feature_column + ['Id']].fillna(-10).set_index('Id')

    del train_data
    gc.collect()

    pos_rate = float(sum(target)) / target.shape[0]
    logger.info('shape %s %s' % data.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), pos_rate))

    all_params = {'max_depth': [6],
                  'n_estimators': [150],
                  'learning_rate': [0.1],
                  'scale_pos_weight': [1],
                  'min_child_weight': [0.01],
                  'subsample': [1],
                  'colsample_bytree': [0.5],
                  'booster': ['dart'],
                  'normalize_type': ['forest'],
                  'sample_type': ['weighted'],
                  'rate_drop': [0.1],
                  'skip_drop': [0.5],
                  'silent': [False],
                  'objective': ['binary:logistic']
                  }

    cv = make_cv()  # cv = StratifiedKFold(target, n_folds=3, shuffle=True, random_state=0)

    from xgboost import DMatrix, train
    logger.info('cv_start')
    with open('train_feature_2_1.py', 'w') as f:
        f.write("LIST_TRAIN_COL = ['" + "', '".join(feature_column) + "']\n\n")

    avg_ntree = 0

    for params in ParameterGrid(all_params):
        logger.info('param: %s' % (params))
        all_ans = None
        all_target = None
        all_ids = None

        for train_idx, test_idx in list(cv):
            list_estimator = []
            ans = []
            insample_ans = []
            for i in [0, 1, 2, 3, '']:  #
                logger.info('model: %s' % i)
                cols = [col for col in feature_column if 'L%s' % i in col]
                train_dmatrix = DMatrix(data.ix[train_idx, cols], label=target.ix[train_idx].values)
                test_dmatrix = DMatrix(data.ix[test_idx, cols], label=target.ix[test_idx].values)
                logger.info('model xg: %s' % i)
                booster = train(params, train_dmatrix,
                                num_boost_round=100,
                                verbose_eval=True)
                """
                booster = train(params, train_dmatrix,
                        evals=[(test_dmatrix, 'eval')],
                        feval=evalmcc_xgb_min,
                        num_boost_round=700,
                        early_stopping_rounds=200,
                        verbose_eval=True)
                """
                avg_ntree += booster.best_ntree_limit
                ans = booster.predict(test_dmatrix,
                                      ntree_limit=booster.best_ntree_limit)
                tree_limit = booster.best_ntree_limit
                score = roc_auc_score(target.ix[test_idx].values, ans)
                logger.info('score: %s' % score)
                logger.info('tree: %s' % tree_limit)
                logger.info('model thresh: %s, score: %s' % mcc_optimize(ans, target.ix[test_idx].values))
                logger.info('train_end')
                if all_ans is None:
                    all_ans = ans
                    all_target = target[test_idx]
                    all_ids = data.ix[test_idx].index.values.astype(int)
                else:
                    all_ans = numpy.r_[all_ans, ans]
                    all_target = numpy.r_[all_target, target[test_idx]]
                    all_ids = numpy.r_[all_ids, data.ix[test_idx].index.values.astype(int)]

                ans = booster.predict(test_dmatrix,
                                      ntree_limit=booster.best_iteration - 10)
                logger.info('model thresh: %s, score: %s' % mcc_optimize(ans, target.ix[test_idx].values))

                ans = booster.predict(test_dmatrix,
                                      ntree_limit=booster.best_iteration + 1)
                logger.info('model thresh: %s, score: %s' % mcc_optimize(ans, target.ix[test_idx].values))
                gc.collect()
                #imp = pandas.DataFrame(get_feature_importance(booster), index=cols, columns=['imp'])
                #_feature_column = list(imp[imp['imp'] == 0].index.values)
                # with open('feature_1026_all.py', 'a') as f:
                #    f.write("LIST_ZERO_COL_ALL = ['" + "', '".join(_feature_column) + "']\n\n")

    pandas.DataFrame(numpy.array([all_ids, all_target, all_ans]).T,
                     columns=['Id', TARGET_COLUMN_NAME, 'pred']).to_csv('stack_1_pred.csv', index=False)

    train_dmatrix = DMatrix(data, label=target.values)
    test_dmatrix = DMatrix(data, label=target.values)
    del data
    gc.collect()
    logger.info('last model xg')
    for params in ParameterGrid(all_params):
        pass
    booster = train(params, train_dmatrix,
                    num_boost_round=500,  # int(avg_ntree / 3) + 1,
                    verbose_eval=True)
    logger.info('end')
    with open('xgb_model_1.pkl', 'wb') as f:
        pickle.dump(booster, f, -1)

    imp = pandas.DataFrame(get_feature_importance(booster), index=feature_column, columns=['imp'])
    _feature_column = list(imp[imp['imp'] == 0].index.values)
    # with open('feature_1026_all.py', 'a') as f:
    #    f.write("LIST_ZERO_COL_ALL2 = ['" + "', '".join(_feature_column) + "']\n\n")
