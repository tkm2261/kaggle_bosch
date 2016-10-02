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
import tensorflow as tf

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_etl_sampling.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

from utils import mcc_optimize, evalmcc_xgb_min
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE
from feature_orig import LIST_COLUMN_NUM
from feature_0928 import LIST_COLUMN_ZERO
#from omit_pos_id import LIST_OMIT_POS_ID, LIST_OMIT_POS_MIN
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

def merge(df, feature_column):
    ids = pandas.read_csv('stack_1_id_1.csv')['0'].values
    data = pandas.read_csv('stack_1_data_1.csv').values
    mst = pandas.DataFrame(data, columns=['L_stack_m%s'%i for i in range(data.shape[1])], index=ids)
    feature_column += ['L_stack_m%s'%i for i in range(data.shape[1])]
    return df.merge(mst, how='left', left_on='Id', right_index=True, copy=False), feature_column


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

    target = train_data[TARGET_COLUMN_NAME].values.astype(numpy.bool_)
    data = train_data[feature_column].fillna(-10).values

    pos_rate = float(sum(target)) / target.shape[0]
    logger.info('shape %s %s' % data.shape)
    logger.info('pos num: %s, pos rate: %s' % (sum(target), pos_rate))
    cv = StratifiedKFold(target, n_folds=3, shuffle=True, random_state=0)
    all_ans = None
    all_target = None
    all_ids = None

    with open('train_feature_1.py', 'w') as f:
        f.write("LIST_TRAIN_COL = ['" + "', '".join(feature_column) + "']\n\n")
    
    sess = tf.InteractiveSession()

    # Create the model
    x = tf.placeholder(tf.float32, [None, data.shape[1]])
    W = tf.Variable(tf.zeros([data.shape[1], 2000]))
    b = tf.Variable(tf.zeros([2000]))
    h1 = tf.nn.relu(tf.matmul(x, W) + b)

    W2 = tf.Variable(tf.zeros([2000, 1000]))
    b2 = tf.Variable(tf.zeros([1000]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W3 = tf.Variable(tf.zeros([1000, 500]))
    b3 = tf.Variable(tf.zeros([500]))
    h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)

    W4 = tf.Variable(tf.zeros([500, 200]))
    b4 = tf.Variable(tf.zeros([200]))
    h4 = tf.nn.relu(tf.matmul(h3, W4) + b4)
    
    W5 = tf.Variable(tf.zeros([200, 100]))
    b5 = tf.Variable(tf.zeros([100]))
    h5 = tf.nn.relu(tf.matmul(h4, W5) + b5)

    W6 = tf.Variable(tf.zeros([100, 10]))
    b6 = tf.Variable(tf.zeros([10]))
    h6 = tf.nn.relu(tf.matmul(h5, W6) + b6)

    W7 = tf.Variable(tf.zeros([10, 1]))
    b7 = tf.Variable(tf.zeros([1]))
    h7 = tf.matmul(h6, W7) + b7
    #h7_drop = tf.nn.dropout(h7, 0.5)
    y = tf.sigmoid(h7)


    y_ = tf.placeholder(tf.float32, [None, 1])
    x_entropy = -1 * y_ * tf.log(y) - (1 - y_) * tf.log(1 - y) 
    loss = tf.reduce_mean(x_entropy, name='xentropy_mean')
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    tf.initialize_all_variables().run()

    logger.info('cv_start')
    for train_idx, test_idx in list(cv)[:1]:
        train_data = data[train_idx]
        train_target = target[train_idx]
        test_data = data[test_idx]
        test_target = target[test_idx]
        logger.info('slice')
        for ep in range(100):
            n_iter = 5
            batchs = StratifiedKFold(train_target, n_folds=n_iter, shuffle=True, random_state=ep)
            avg_cost = 0.
            logger.info('epock: %s'%ep)
            for i, (_, batch_idx) in enumerate(batchs):
                logger.info(' batch: %s/%s'%(i + 1, n_iter))
                batch_xs = train_data[batch_idx]
                batch_ys = train_target[batch_idx]
                _, cost = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys.reshape(-1, 1)})
                #train_step.run({x: batch_xs, y_: batch_ys.reshape(-1, 1)})
                avg_cost += cost / n_iter
            logger.info('loss: %s'%avg_cost)
            pred = y.eval({x: test_data}, session=sess).reshape((1, -1))[0]
            print(pred)
            score = mcc_optimize(pred, test_target)
            logger.info('score: %s %s' % score)
            score = roc_auc_score(test_target, pred)
            logger.info('suc: %s' % score)


        pred = y.eval({x: test_data}, session=sess).reshape((1, -1))[0]

        score = mcc_optimize(pred, test_target)
        logger.info('score: %s %s' % score)
        score = roc_auc_score(test_target, pred)
        logger.info('suc: %s' % score)
