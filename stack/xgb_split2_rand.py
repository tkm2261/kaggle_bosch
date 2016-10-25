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
    ids = pandas.read_csv('stack_1_id_1.csv.gz')['0'].values
    data = pandas.read_csv('stack_1_data_1.csv.gz')

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


def make_chi_all(df, feature_columns):
    for num in [6000, 7000, 8000, 9000]:
        logger.info('CHI!! %s' % num)
        data = pandas.read_csv('../protos/test_chi_all_%s_sle.csv.gz' % num)
        new_cols = ['L_chi_%s_%s' % (num, col) if col != 'Id' else col for col in data.columns.values]
        data.columns = new_cols
        feature_columns += [col for col in new_cols if col != 'Id']
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
    train_data = pandas.concat(p.map(read_csv2,
                                     glob.glob('../data/train_etl/*')
                                     #['../data/train_join/train_join_%s.csv.gz' % i for i in range(119)]
                                     )).reset_index(drop=True)

    p.close()
    p.join()
    feature_column = [col for col in train_data.columns if col != TARGET_COLUMN_NAME and col != 'Id']
    logger.info('load end')

    train_data, feature_column = make_stack(train_data, feature_column)
    train_data, feature_column = make_chi(train_data, feature_column)
    train_data, feature_column = make_chi2(train_data, feature_column)
    train_data, feature_column = make_chi3(train_data, feature_column)
    #train_data, feature_column = make_chi_all(train_data, feature_column)

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

    # data_valid, target_valid = get_sample_data(feature_column)

    pos_rate = float(sum(target)) / target.shape[0]
    logger.info('shape %s %s' % data.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), pos_rate))

   # param: {'max_depth': 9, 'subsample': 1, 'reg_alpha': 0.01,
   # 'min_child_weight': 0.1, 'scale_pos_weight': 1, 'gamma': 0.4,
   # 'n_estimators': 60, 'learning_rate': 0.1, 'colsample_bytree': 0.6}
    all_params = {'max_depth': [9],
                  'n_estimators': [20],  # 31
                  'learning_rate': [0.1],
                  'scale_pos_weight': [1],
                  'min_child_weight': [0.1],
                  'subsample': [1],
                  'colsample_bytree': [0.6],
                  'reg_alpha': [0.1],
                  'gamma': [0.4]
                  }

    cv = StratifiedKFold(target, n_folds=3, shuffle=True, random_state=0)

    omit_idx = ids[~ids.isin(LIST_OMIT_POS_ID)].index.values
    logger.info('cv_start')
    for params in ParameterGrid(all_params):
        pass
    import random
    random.seed(0)

    base_score = -1
    base_score_mcc = -1
    map_test = {}
    while 1:
        all_ans = None
        all_target = None
        all_ids = None
        data = data[feature_column]
        logger.info('shape %s %s' % data.shape)
        with open('train_feature_2_rand.py', 'w') as f:
            f.write("LIST_TRAIN_COL = ['" + "', '".join(feature_column) + "']\n\n")

        if base_score > 0:
            cand = [col for col in feature_column if 'pred' not in col and '-' not in col]

            for _ in range(100):
                merge_cols = sorted(random.sample(cand, 3))
                new_col = '-'.join(merge_cols)
                if new_col in map_test:
                    continue
                else:
                    map_test[new_col] = None

                df_hash = data[merge_cols].astype(str).sum(axis=1)
                df_hash = pandas.DataFrame(df_hash, columns=['hash'])
                df_cnt = df_hash.groupby('hash')[['hash']].count()

                df_cnt.columns = [new_col]
                df_hash = df_hash.merge(df_cnt, how='left', left_on='hash', right_index=True)[new_col].values
                data[new_col] = df_hash

        for train_idx, test_idx in list(cv)[2:]:
            train_omit_idx = numpy.intersect1d(train_idx, omit_idx)
            logger.info('ommit size: %s %s' % (train_idx.shape[0], len(train_omit_idx)))

            ans = []
            insample_ans = []
            for i in ['']:  #
                logger.info('model: %s' % i)
                cols = [col for col in data.columns.values if 'L%s' % i in col]
                logger.info('model xg: %s' % i)
                model = XGBClassifier(seed=0)
                gc.collect()
                model.set_params(**params)
                if 1:
                    model.fit(data.ix[train_idx, cols], target[train_idx])
                else:
                    model.fit(data.ix[train_idx, cols], target[train_idx],
                              eval_set=[(data.ix[test_idx, cols], target[test_idx])],
                              eval_metric=evalmcc_xgb_min,
                              early_stopping_rounds=1000,
                              verbose=True)

                ans = model.predict_proba(data.ix[test_idx, cols])[:, 1]
                score = roc_auc_score(target[test_idx], ans)
                thresh, mcc = mcc_optimize(ans, target[test_idx])
                logger.info('auc: %s thresh: %s, score: %s' % (score, thresh, mcc))

                """
                for t in range(1, 101):
                    ans = model.predict_proba(data.ix[test_idx, cols], ntree_limit=t)[:, 1]
                    score = roc_auc_score(target[test_idx], ans)
                    logger.info('    score: %s' % score)
                    logger.info('    model thresh: %s, score: %s' % mcc_optimize(ans, target[test_idx]))
                """
            logger.info('train_end')
            if all_ans is None:
                all_ans = ans
                all_target = target[test_idx]
                all_ids = ids.ix[test_idx].values
            else:
                all_ans = numpy.r_[all_ans, ans]
                all_target = numpy.r_[all_target, target[test_idx]]
                all_ids = numpy.r_[all_ids, ids.ix[test_idx]]

            if base_score == -1:
                base_score = score
                base_score_mcc = mcc

            if score > base_score or mcc > base_score_mcc:
                logger.info('auc improving %s' % (score - base_score))
                logger.info('mcc improving %s' % (mcc - base_score_mcc))
                feature_column.append(new_col)
                base_score = max(base_score, score)
                base_score_mcc = max(base_score_mcc, mcc)

                imp = pandas.DataFrame(model.feature_importances_, index=cols, columns=['imp'])
                feature_column = list(imp[imp['imp'] != 0].index.values)

        score = roc_auc_score(all_target, all_ans)
        logger.info('score: %s' % score)
        score = log_loss(all_target, all_ans)
        logger.info('logloss score: %s' % score)

        logger.info('cv model thresh: %s, score: %s' % mcc_optimize(all_ans, all_target))

    for i in ['']:
        logger.info('model: %s' % i)
        cols = [col for col in feature_column if 'L%s' % i in col]
        logger.info('model xg: %s' % i)
        model = XGBClassifier(seed=0)
        model.set_params(**params)
        model.fit(data[cols], target)
        """
        model.fit(data[cols], target,
                  # eval_set=[(data_valid, target_valid)],
                  eval_metric=evalmcc_xgb_min,
                  early_stopping_rounds=50,
                  verbose=True)
        """

    with open('tmp_model.pkl', 'wb') as f:
        pickle.dump(model, f, -1)

    ids = pandas.read_csv('stack_1_id_2.csv')['0'].values
    _data = pandas.read_csv('stack_1_data_2.csv')
    logger.info('old data %s %s' % _data.shape)
    df = pandas.Series(all_ans, index=all_ids)
    logger.info('df data %s' % df.shape[0])

    _data[data.columns.values[-1]] = df[ids].values
    _data.to_csv('stack_1_data_2_2.csv', index=False)

    with open('list_xgb_model_2.pkl', 'rb') as f:
        list_model = pickle.load(f)

    list_model[-2] = model
    with open('list_xgb_model_2_1.pkl', 'wb') as f:
        pickle.dump(list_model, f, -1)
