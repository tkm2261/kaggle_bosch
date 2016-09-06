# encoding: utf-8
import os
import logging
import pandas
import pickle
from xgboost import XGBClassifier

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT)
TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
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
    train_data = pandas.read_csv(TRAIN_DATA)
    train_data = train_data.fillna(0)
    logger.info('load end')
    logger.info('shape %s %s' % train_data.shape)

    feature_column = [col for col in train_data.columns.values if col != TARGET_COLUMN_NAME]
    target = train_data[TARGET_COLUMN_NAME]
    data = train_data[feature_column]

    logger.info('shape %s %s' % data.shape)

    params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.5,
              'max_depth': 5, 'min_child_weight': 0.01}
    model = XGBClassifier(seed=0)
    model.set_params(**params)
    model.fit(data, target)

    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f, -1)
