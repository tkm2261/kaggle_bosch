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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC


APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from utils import mcc_optimize, evalmcc_xgb_min
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM
from feature_0928 import LIST_COLUMN_ZERO
from feature_1009 import LIST_COLUMN_ZERO_MIX
from omit_pos_id import LIST_OMIT_POS_ID, LIST_OMIT_POS_MIN

from train_feature_1 import LIST_TRAIN_COL

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

if __name__ == '__main__':
    logger.info('load start')
    """
    p = Pool()

    train_data = pandas.concat(p.map(read_csv,
                                     glob.glob(os.path.join(DATA_DIR, 'train_etl/*'))
                                     )).reset_index(drop=True)
    train_data_cnt = pandas.concat(p.map(read_df,
                                         pandas.read_csv(os.path.join(
                                             DATA_DIR, 'train_etl2/train.csv.gz'), chunksize=10000),
                                         )).reset_index(drop=True)
    p.close()
    p.join()
    logger.info('shape %s %s' % train_data.shape)

    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']
    feature_column_cnt = [col for col in train_data_cnt.columns if col != TARGET_COLUMN_NAME and col != 'Id']

    train_data_cnt = train_data_cnt[[col for col in train_data_cnt.columns.values if col != TARGET_COLUMN_NAME]]
    train_data = train_data[['Id', TARGET_COLUMN_NAME] + feature_column]
    train_data = train_data.merge(train_data_cnt, how='left', left_on='Id', right_on='Id', copy=False)

    if 1:
        train_data.to_csv(os.path.join(DATA_DIR, 'train_join.csv.gz'), compression='gzip', index=False)
 
    del train_data_cnt
    gc.collect()
    """
    p = Pool()
    train_data = pandas.concat(p.map(read_df,
                                     pandas.read_csv(os.path.join(
                                         DATA_DIR, 'train_join.csv.gz'), chunksize=10000)
                                     ), ignore_index=False).reset_index(drop=True)
    p.close()
    p.join()
    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']
    logger.info('load end')

    train_data, feature_column = make_stack(train_data, feature_column)
    #feature_column += feature_column_cnt
    feature_column = [col for col in feature_column if col not in LIST_COLUMN_ZERO_MIX]
    feature_column = [col for col in feature_column if 'hash' not in col or 'cnt' in col]

    target = train_data[TARGET_COLUMN_NAME].values.astype(numpy.bool_)
    data = train_data[feature_column].fillna(-10)
    ids = train_data['Id']

    del train_data
    gc.collect()

    pos_rate = float(sum(target)) / target.shape[0]
    logger.info('shape %s %s' % data.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), pos_rate))

    all_params = {'max_depth': [11],
                  'n_estimators': [200],
                  'learning_rate': [0.1],
                  'scale_pos_weight': [1],
                  'min_child_weight': [0.01],
                  'subsample': [1],
                  'colsample_bytree': [0.5],
                  'reg_alpha': [0.1],
                  }

    cv = StratifiedKFold(target, n_folds=3, shuffle=True, random_state=0)
    all_ans = None
    all_target = None
    all_ids = None

    omit_idx = ids[~ids.isin(LIST_OMIT_POS_ID)].index.values
    with open('train_feature_2.py', 'w') as f:
        f.write("LIST_TRAIN_COL = ['" + "', '".join(feature_column) + "']\n\n")
    logger.info('cv_start')
    for params in ParameterGrid(all_params):
        logger.info('param: %s' % (params))
        for train_idx, test_idx in list(cv):
            train_omit_idx = numpy.intersect1d(train_idx, omit_idx)
            logger.info('ommit size: %s %s' % (train_idx.shape[0], len(train_omit_idx)))
            list_estimator = []
            ans = []
            insample_ans = []
            for i in [0, 1, 2, 3, '']:  #
                logger.info('model: %s' % i)
                cols = [col for col in feature_column if 'L%s' % i in col]

                logger.info('model lg: %s' % i)
                model = SGDClassifier(loss='log', penalty='l1', n_iter=20, random_state=0, n_jobs=-1)
                model.fit(data.ix[train_idx, cols], target[train_idx])
                list_estimator.append(model)
                ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
                insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])

                logger.info('model svc: %s' % i)
                model = SGDClassifier(loss='hinge', penalty='l1', n_iter=20, random_state=0, n_jobs=-1)
                model.fit(data.ix[train_idx, cols], target[train_idx])
                list_estimator.append(model)
                ans.append(sigmoid(model.decision_function(data.ix[test_idx, cols])))
                insample_ans.append(model.decision_function(data.ix[train_idx, cols]))

                logger.info('model rf: %s' % i)
                model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0)
                gc.collect()
                model.fit(data.ix[train_idx, cols], target[train_idx])
                list_estimator.append(model)
                ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
                insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])

                logger.info('model rf2: %s' % i)
                model = RandomForestClassifier(n_estimators=100,
                                               min_samples_leaf=10,
                                               n_jobs=-1,
                                               random_state=10)
                gc.collect()
                model.fit(data.ix[train_idx, cols], target[train_idx])
                list_estimator.append(model)
                ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
                insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])

                logger.info('model et: %s' % i)
                model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=-1)
                model.fit(data.ix[train_idx, cols], target[train_idx])
                list_estimator.append(model)
                ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
                insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])

                logger.info('model et2: %s' % i)
                model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=10, random_state=10, n_jobs=-1)
                model.fit(data.ix[train_idx, cols], target[train_idx])
                list_estimator.append(model)
                ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
                insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])

                logger.info('model nb: %s' % i)
                model = GaussianNB()
                gc.collect()
                model.fit(data.ix[train_idx, cols], target[train_idx])
                list_estimator.append(model)
                ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
                insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])

                logger.info('model xg: %s' % i)
                model = XGBClassifier(seed=0)
                gc.collect()
                model.set_params(**params)
                model.fit(data.ix[train_idx, cols], target[train_idx],
                          eval_metric=evalmcc_xgb_min,
                          verbose=False)
                list_estimator.append(model)
                ans.append(model.predict_proba(data.ix[test_idx, cols])[:, 1])
                insample_ans.append(model.predict_proba(data.ix[train_idx, cols])[:, 1])

            with open('list_xgb_model_2.pkl', 'wb') as f:
                pickle.dump(list_estimator, f, -1)

            logger.info('train_end')
            ans = numpy.array(ans).T
            insample_ans = numpy.array(insample_ans).T
            if all_ans is None:
                all_ans = ans
                all_target = target[test_idx]
                all_ids = ids.ix[test_idx].values
            else:
                all_ans = numpy.r_[all_ans, ans]
                all_target = numpy.r_[all_target, target[test_idx]]
                all_ids = numpy.r_[all_ids, ids.ix[test_idx]]

            model = XGBClassifier(seed=0)
            model.fit(ans, target[test_idx])
            pred = model.predict_proba(ans)[:, 1]
            logger.info('model thresh: %s, score: %s' % mcc_optimize(pred, target[test_idx]))
            pred = ans.max(axis=1)
            logger.info('max thresh: %s, score: %s' % mcc_optimize(pred, target[test_idx]))
            pred = ans.min(axis=1)
            logger.info('min thresh: %s, score: %s' % mcc_optimize(pred, target[test_idx]))

            logger.info('mean thresh: %s, score: %s' % mcc_optimize(ans.mean(axis=1), target[test_idx]))

            for j in range(ans.shape[1]):
                score = roc_auc_score(target[test_idx], ans[:, j])
                logger.info('score: %s' % score)
                logger.info('model thresh: %s, score: %s' % mcc_optimize(ans[:, j], target[test_idx]))

            score = roc_auc_score(target[test_idx], pred)
            logger.info('INSAMPLE score: %s' % score)
            pred = model.predict_proba(insample_ans)[:, 1]  # ans.max(axis=1)
            score = roc_auc_score(target[train_idx], pred)
            logger.info('INSAMPLE train score: %s' % score)

            list_estimator.append(model)

    pandas.DataFrame(all_ans).to_csv('stack_1_data_2.csv', index=False)
    pandas.DataFrame(all_target).to_csv('stack_1_target_2.csv', index=False)
    pandas.DataFrame(all_ids).to_csv('stack_1_id_2.csv', index=False)

    idx = 0
    for i in [0, 1, 2, 3, '']:
        logger.info('model: %s' % i)
        cols = [col for col in feature_column if 'L%s' % i in col]

        gc.collect()
        logger.info('model lg: %s' % i)
        model = SGDClassifier(loss='log', penalty='l1', n_iter=20, random_state=0, n_jobs=-1)
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

        logger.info('model svc: %s' % i)
        model = SGDClassifier(loss='hinge', penalty='l1', n_iter=20, random_state=0, n_jobs=-1)
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

        logger.info('model rf: %s' % i)
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0)
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

        logger.info('model rf2: %s' % i)
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=-1, random_state=10)
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

        logger.info('model et: %s' % i)
        model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=-1)
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

        logger.info('model et2: %s' % i)
        model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=10, random_state=10, n_jobs=-1)
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

        logger.info('model nb: %s' % i)
        model = GaussianNB()
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

        logger.info('model xg: %s' % i)
        model = XGBClassifier(seed=0)
        model.set_params(**params)
        model.fit(data[cols], target)
        list_estimator[idx] = model
        idx += 1

    with open('list_xgb_model_2.pkl', 'wb') as f:
        pickle.dump(list_estimator, f, -1)
