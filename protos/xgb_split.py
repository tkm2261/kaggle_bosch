# encoding: utf-8
import os
import logging
import pandas
import pickle
import numpy
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.cross_validation import StratifiedKFold

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')

TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'
from feature import LIST_FEATURE_COLUMN_NAME
log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')
logger = logging.getLogger(__name__)

def mcc(y_true, y_pred):
    n = y_true.shape[0]
    true_pos = sum(1. for i in range(n) if y_true[i] == 1 and y_pred[i] == 1)
    true_neg = sum(1. for i in range(n) if y_true[i] == 0 and y_pred[i] == 0)
    false_pos = sum(1. for i in range(n) if y_true[i] == 0 and y_pred[i] == 1)
    false_neg = sum(1. for i in range(n) if y_true[i] == 1 and y_pred[i] == 0)    
    """
    s = (true_pos + false_neg) / n
    p = (true_pos + false_pos) / n

    a = true_pos / n - s * p
    b = numpy.sqrt(s * p * (1 - s) * (1 - p))
    """
    a = true_pos * true_neg - false_pos * false_neg
    b = (true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg)
    return a / numpy.sqrt(b)
    
def mcc_scoring(estimator, X, y):

    y_pred_prb = estimator.predict_proba(X)[:, 1]
    list_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    max_score = -1
    for thresh in list_thresh:
        y_pred = numpy.where(y_pred_prb >= thresh, 1, 0)
        score = mcc(y, y_pred)
        logger.info('thresh: %s, score: %s'%(thresh, score))
        if score > max_score:
            max_score = score
    return max_score


if __name__ == '__main__':
    logger.info('load start')
    train_data = pandas.read_csv('pos_data_170.csv.gz', index_col=0)
    train_data = train_data.fillna(-1)
    logger.info('load end')
    logger.info('shape %s %s' % train_data.shape)

    #feature_column = [col for col in train_data.columns.values if col != TARGET_COLUMN_NAME]
    target = pandas.read_csv('pos_target_170.csv.gz', header=None).ix[:, 1].values
    data = train_data[LIST_FEATURE_COLUMN_NAME]

    logger.info('shape %s' % target.shape)
    logger.info('pos num: %s, pos rate: %s'%(sum(target), float(sum(target)) / target.shape[0]))
    
    params = {'subsample': 1, 'learning_rate': 0.1, 'colsample_bytree': 0.3,
              'max_depth': 5, 'min_child_weight': 0.01, 'n_estimators': 200,
              'scale_pos_weight': 10}
    list_estimator =[]
    cv = StratifiedKFold(target, n_folds=10, shuffle=True, random_state=0)

    for train_idx, test_idx in list(cv)[:1]:
        ans = []
        insample_ans = []
        for i in list(range(4)) + ['']:
            cols = [col for col in LIST_FEATURE_COLUMN_NAME if 'L%s'%i in col]
            model = XGBClassifier(seed=0)
            model.set_params(**params)
            model.fit(data[cols].values[train_idx], target[train_idx])
            list_estimator.append(model)
            ans.append(model.predict_proba(data[cols].values[test_idx])[:, 1])
            insample_ans.append(model.predict_proba(data[cols].values[train_idx])[:, 1])            
        ans = numpy.array(ans).T
        insample_ans = numpy.array(insample_ans).T
        #model = LogisticRegression(n_jobs=-1, class_weight='balanced')
        model = LogisticRegressionCV(n_jobs=-1, class_weight='balanced', scoring='roc_auc')
        #model = XGBClassifier(seed=0)
        #model.set_params(**params)
        model.fit(ans, target[test_idx])
        pred = model.predict_proba(ans)[:, 1] # ans.max(axis=1)
        score = roc_auc_score(target[test_idx], pred)
        logger.info('INSAMPLE score: %s' % score)
        pred = model.predict_proba(insample_ans)[:, 1] # ans.max(axis=1)
        score = roc_auc_score(target[train_idx], pred)
        logger.info('INSAMPLE train score: %s' % score)
        
        list_estimator.append(model)

    for i, ii in enumerate(list(range(4)) + ['']):
        cols = [col for col in LIST_FEATURE_COLUMN_NAME if 'L%s'%ii in col]
        model = XGBClassifier(seed=0)
        model.set_params(**params)
        model.fit(data[cols].values, target)
        list_estimator[i] = model 


    """
    params = {'max_depth': [3, 5, 10],
              'learning_rate': [0.01, 0.1, 1],
              'min_child_weight': [0.01, 0.1, 1],
              'subsample': [0.1, 0.5, 1],
              'colsample_bytree': [0.3, 0.5, 1],
    }

    params = {'max_depth': [3],
              'learning_rate': [0.1],
              'min_child_weight': [0.01],
              'subsample': [1],
              'colsample_bytree': [0.3],
              'n_estimators': [200],
              }
    
    cv = GridSearchCV(model,
                      params,
                      scoring='roc_auc',
                      n_jobs=1,
                      refit=False,
                      verbose=10
                      #scoring=mcc_scoring,
    )
    cv.fit(data, target)
    logger.info('best param: %s'%cv.best_params_)
    logger.info('best score: %s'%cv.best_score_)
    
    model.set_params(**cv.best_params_)
    model.fit(data, target)

    score = roc_auc_score(target, model.predict_proba(data)[:, 1])
    logger.info('INSAMPLE score: %s' % score)
    """    
    with open('list_xgb_model.pkl', 'wb') as f:
        pickle.dump(list_estimator, f, -1)
