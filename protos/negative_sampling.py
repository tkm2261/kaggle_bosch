# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
import gc
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TRAIN_POSITIVE_DATA = os.path.join(DATA_DIR, 'train_simple_join_pos.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'
from feature import LIST_FEATURE_COLUMN_NAME
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('load start')

    train_pos_data = pandas.read_csv(TRAIN_POSITIVE_DATA)
    train_pos_data = train_pos_data.fillna(-1)
    list_train_data = pandas.read_csv(TRAIN_DATA, chunksize=train_pos_data.shape[0])

    logger.info('load end')
    logger.info('pos_data shape %s %s' % train_pos_data.shape)

    pos_target = train_pos_data[TARGET_COLUMN_NAME]
    pos_data = train_pos_data[LIST_FEATURE_COLUMN_NAME]

    start = 0
    if 1:
        pos_data = pandas.read_csv('pos_data_%s.csv.gz' % (170), index_col=0)
        pos_target = pandas.read_csv('pos_target_%s.csv.gz' % (170), header=None, index_col=0)[1]

    params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.5,
              'max_depth': 5, 'min_child_weight': 0.01}
    #model = LogisticRegression(n_jobs=-1)
    model = XGBClassifier(seed=0)
    model.set_params(**params)
    model.fit(pos_data, pos_target)

    for i, train_data in enumerate(list_train_data):
        if i < start:
            del train_data
            gc.collect()
            continue
        # elif i == start:
        #   model = LogisticRegression(n_jobs=-1)
        #   model.fit(pos_data, pos_target)

        train_data = train_data.fillna(-1)

        if 0:  # i == 0:
            pos_target = pos_target.append(train_data[TARGET_COLUMN_NAME])
            pos_data = pos_data.append(train_data[LIST_FEATURE_COLUMN_NAME])
        else:
            target = train_data[TARGET_COLUMN_NAME]
            data = train_data[LIST_FEATURE_COLUMN_NAME]
            neg_target = target[target == 0]
            neg_data = data[target == 0]
            score = model.predict_proba(neg_data)[:, 1]
            thresh = float(sum(pos_target)) / len(pos_target)
            idx = numpy.argsort(score)[::-1][:1000]
            idx = [neg_data.index.values[ix] for ix in idx if score[ix] > thresh]
            pos_target = pos_target.append(neg_target.ix[idx])
            pos_data = pos_data.append(neg_data.ix[idx, :])

        logger.info('%s: pos shape %s' % (i, pos_data.shape[0]))

        score = roc_auc_score(pos_target, model.predict_proba(pos_data)[:, 1])
        logger.info('INSAMPLE score: %s' % score)

        if (i + 1) % 10 == 0:
            pos_data.to_csv('pos_data_170_%s.csv' % (i + 1))
            pos_target.to_csv('pos_target_170_%s.csv' % (i + 1))
        del train_data
        gc.collect()
