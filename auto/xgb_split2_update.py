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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, log_loss
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
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
from feature_1015 import LIST_ZEOO_2

from feature_1019 import LIST_ZERO_STACK_2, LIST_ZERO_STACK_2_2
from feature_1020 import LIST_CHI_STACK_3
from feature_1021 import LIST_ZERO_STACK_3
from omit_pos_id import LIST_OMIT_POS_ID, LIST_OMIT_POS_MIN

from train_feature_1 import LIST_TRAIN_COL


from feature_1018 import LIST_COLUMN_MCC
from feature_1017 import LIST_MCC_UP


log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='INFO')
logger = logging.getLogger(__name__)

from autosklearn.classification import AutoSklearnClassifier

import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification
from operator import itemgetter


def report(grid_scores, n_top=3):

    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            numpy.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def hash_groupby(df, feature_column):

    new_feature_column = list(feature_column)
    for col in feature_column:
        if 'hash' not in col:
            continue
        logger.info('hash col%s' % col)
        tmp = df.groupby(col)[[TARGET_COLUMN_NAME]].count()
        # tmp_mean = tmp[TARGET_COLUMN_NAME].mean()
        # tmp[TARGET_COLUMN_NAME][tmp[TARGET_COLUMN_NAME] < 2] = 2
        tmp.columns = [col + '_prob']
        new_feature_column.append(col + '_prob')
        df = pandas.merge(df, tmp, how='left', left_on=col, right_index=True)

    df[[col for col in new_feature_column if 'hash' in col]].to_csv('hash_prob.csv', index=True)
    return df, new_feature_column


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
    ids = pandas.read_csv('../stack/stack_1_id_1.csv.gz')['0'].values
    data = pandas.read_csv('../stack/stack_1_data_1.csv.gz')

    new_cols = ['L0_L1_L2_L3_pred_%s' % col for col in data.columns.values]
    data.columns = new_cols
    feature_columns += new_cols

    data['Id'] = ids
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')
    return df, feature_columns


def make_chi(df, feature_columns):
    logger.info('CHI!!')
    """
    data = pandas.read_csv('../protos/train_chi_all_1000.csv.gz')
    new_cols = ['L_chi_%s' % col for col in data.columns.values]
    data.columns = new_cols
    """

    data = pandas.read_csv('../data/train_chi_1000.csv.gz')
    feature_columns += [col for col in data.columns.values if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns


def make_chi2(df, feature_columns):
    logger.info('CHI!!')
    """
    data = pandas.read_csv('../protos/train_chi_all_1000.csv.gz')
    new_cols = ['L_chi_%s' % col for col in data.columns.values]
    data.columns = new_cols
    """

    data = pandas.read_csv('../data/train_chi_2000.csv.gz')
    feature_columns += [col for col in data.columns.values if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns


def make_chi3(df, feature_columns):
    logger.info('CHI!!')
    data = pandas.read_csv('../data/train_chi_3000.csv.gz')
    feature_columns += [col for col in data.columns.values if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns


"""
def make_chi3(df, feature_columns):
    logger.info('CHI!!')

    data = pandas.read_csv('../protos/train_chi_all_3000.csv.gz')
    new_cols = ['L_chi_%s' % col if col != 'Id' else col for col in data.columns.values]
    data.columns = new_cols
    feature_columns += [col for col in new_cols if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')

    return df, feature_columns
"""


def make_magic(df, feature_columns):
    logger.info('make magic')
    # df['L_Id'] = df['Id'].values
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


def read_csv(filename):
    'converts a filename to a pandas dataframe'
    tmp = [col for col in LIST_MCC_UP if 'pred' not in col]
    cols = sorted(list(set(LIST_COLUMN_MCC + tmp)))
    df = pandas.read_csv(filename, usecols=['Id', TARGET_COLUMN_NAME] + cols)
    return df


def read_csv2(path):
    return pandas.read_csv(path)


def get_sample_data(cols):
    paths = os.path.join(DATA_DIR, 'test_right/*')

    p = Pool()
    df = pandas.concat(
        p.map(read_csv2, glob.glob(paths)),
        ignore_index=True)
    p.close()
    p.join()
    feature_columns = [col for col in df.columns.values if col != 'Id' and col != TARGET_COLUMN_NAME]
    logger.info('end sample')
    data = pandas.read_csv('../protos/test_chi_all_1000.csv.gz')
    new_cols = ['L_chi_%s' % col for col in data.columns.values]
    data.columns = new_cols
    feature_columns += [col for col in new_cols if col != 'L_chi_Id']

    df = pandas.merge(df, data, how='left', left_on='Id', right_on='L_chi_Id')

    data = pandas.read_csv('pred_stack_1.csv')
    feature_columns += [col for col in new_cols if col != 'Id']
    df = pandas.merge(df, data, how='left', left_on='Id', right_on='Id')
    logger.info('end test join')
    return df[cols], df[TARGET_COLUMN_NAME].values

if __name__ == '__main__':
    logger.info('load start')

    p = Pool()
    train_data = pandas.concat(p.map(read_csv,
                                     ['../data/train_join/train_join_%s.csv.gz' % i for i in range(119)]
                                     )).reset_index(drop=True)

    p.close()
    p.join()
    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']
    logger.info('load end')

    train_data, feature_column = make_stack(train_data, feature_column)
    train_data, feature_column = make_chi(train_data, feature_column)
    train_data, feature_column = make_chi2(train_data, feature_column)
    train_data, feature_column = make_chi3(train_data, feature_column)

    # feature_column += feature_column_cnt
    # feature_column = [col for col in feature_column if col not in LIST_COLUMN_ZERO_MIX]
    # feature_column = [col for col in feature_column if col not in LIST_ZEOO_2]
    feature_column = [col for col in feature_column if col not in LIST_ZERO_STACK_2]
    feature_column = [col for col in feature_column if col not in LIST_ZERO_STACK_2_2]
    #feature_column = [col for col in feature_column if col not in LIST_CHI_STACK_3]
    feature_column = [col for col in feature_column if col not in LIST_ZERO_STACK_3]
    # feature_column = [col for col in feature_column if 'hash' not in col or 'cnt' in col]

    target = train_data[TARGET_COLUMN_NAME].values.astype(numpy.bool_)
    data = train_data[feature_column].fillna(-10)
    ids = train_data['Id']

    del train_data
    gc.collect()

    model = AutoSklearnClassifier(time_left_for_this_task=20000,
                                  per_run_time_limit=600,
                                  initial_configurations_via_metalearning=25,
                                  ensemble_size=50,
                                  ensemble_nbest=50,
                                  seed=1,
                                  ml_memory_limit=12000,
                                  include_estimators=None,
                                  include_preprocessors=None,
                                  resampling_strategy='holdout',
                                  tmp_folder='./tmp/',
                                  output_folder='./out/',
                                  delete_tmp_folder_after_terminate=False,                                                                                                                     delete_output_folder_after_terminate=False,                                                                                                                  shared_mode=False)

    model.fit(data,
              target,
              metric='f1_metric',
              feat_type=None,
              dataset_name='numerai_20161021')

    try:
        report(model.grid_scores_)
    except:
        pass

    with open('result.txt', 'w') as f:
        f.write(model.show_models())

    cv = StratifiedKFold(target, n_folds=3, shuffle=True, random_state=0)
    for train_idx, test_idx in list(cv)[:1]:
        model.refit(data.ix[train_idx, :], target[train_idx])
        ans = model.predict_proba(data.ix[test_idx, :])[:, 1]
        score = roc_auc_score(target[test_idx], ans)
        print('    score: %s' % score)
        print('    model thresh: %s, score: %s' % mcc_optimize(ans, target[test_idx]))

    model.refit(data.ix, target)
    del data
    gc.collect()

    try:
        with open('tmp_model.pkl', 'wb') as f:
            pickle.dump(model, f, -1)
    except:
        pass
    p = Pool()

    df = pandas.concat(p.map(read_csv,
                             glob.glob(os.path.join(DATA_DIR, 'test_join/*'))
                             )).reset_index(drop=True)

    p.close()
    p.join()

    pred = pandas.read_csv('../stack/ans_stack_1.csv.gz', index_col='Id')
    df = df.merge(pred, how='left', left_on='Id', right_index=True, copy=False)
    df, _ = make_chi(df, list(feature_column))
    df, _ = make_chi2(df, list(feature_column))
    df, _ = make_chi3(df, list(feature_column))

    data = df[feature_column].fillna(-10)
    ids = df['Id']

    predict_proba = model.predict_proba(data)[:, 1]
    ans = pandas.DataFrame(ids)
    ans['Response'] = predict_proba
    ans['proba'] = predict_proba

    ans.to_csv('submit.csv', index=False)
