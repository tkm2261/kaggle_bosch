# encoding: utf-8
import os
import logging
import pandas
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT)
TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TRAIN_POSITIVE_DATA = os.path.join(DATA_DIR, 'train_simple_join_pos.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'
#from feature import LIST_FEATURE_COLUMN_NAME
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('load start')
    train_pos_data = pandas.read_csv(TRAIN_POSITIVE_DATA)
    train_pos_data = train_pos_data.fillna(0)
    list_train_data = pandas.read_csv(TRAIN_DATA, chunksize=train_pos_data.shape[0])
    logger.info('load end')
    logger.info('pos_data shape %s %s' % train_pos_data.shape)

    feature_column = [col for col in train_pos_data.columns.values if col != TARGET_COLUMN_NAME and col != 'Id']
    logger.info('feature_column %s' % len(feature_column))
    pos_target = train_pos_data[TARGET_COLUMN_NAME]
    pos_data = train_pos_data[feature_column]
    params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.5,
              'max_depth': 5, 'min_child_weight': 0.01}

    pos_target = train_pos_data[TARGET_COLUMN_NAME]
    pos_data = train_pos_data[feature_column]

    for i, train_data in enumerate(list_train_data):
        logger.info('pos shape %s %s' % pos_data.shape)
        model = XGBClassifier(seed=0)
        model.set_params(**params)

        if i == 0:
            pos_target = train_pos_data[TARGET_COLUMN_NAME]
            pos_data = train_pos_data[feature_column]

            pos_target = pos_target.append(train_data[TARGET_COLUMN_NAME])
            pos_data = pos_data.append(train_data[feature_column])
        else:
            target = train_data[TARGET_COLUMN_NAME]
            data = train_data[feature_column]
            neg_target = target[target == 0]
            neg_data = data[target == 0]
            score = model.predict_proba(neg_data)[:, 1]

            pos_target = pos_target.append(neg_target[score > 0.5])
            pos_data = pos_data.append(neg_data[score > 0.5])

        model.fit(pos_data, pos_target)
        score = roc_auc_score(pos_target, model.predict_proba(pos_data)[:, :1])
        logger.info('INSAMPLE score: %s' % score)

    pos_data.to_csv('pos_data.csv')
    pos_target.to_csv('pos_target.csv')
