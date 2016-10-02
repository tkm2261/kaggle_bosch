# encoding: utf-8

import pickle
import os
import pandas
import logging
import numpy
import gc
import re
from collections import defaultdict
from feature import LIST_FEATURE_COLUMN_NAME, LIST_DUPLICATE_COL_NAME, LIST_POSITIVE_NA_COL_NAME, LIST_SAME_COL, LIST_DUPLIDATE_CAT, LIST_DUPLIDATE_DATE

APP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATA_DIR = os.path.join(APP_ROOT, 'data')
TRAIN_DATA = os.path.join(DATA_DIR, 'train_simple_join.csv.gz')
TRAIN_POSITIVE_DATA = os.path.join(DATA_DIR, 'train_simple_join_pos.csv.gz')
TEST_DATA = os.path.join(DATA_DIR, 'test_simple_join.csv.gz')
TARGET_COLUMN_NAME = u'Response'

log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
logging.basicConfig(format=log_fmt,
                    datefmt='%Y-%m-%d/%H:%M:%S',
                    level='DEBUG')
logger = logging.getLogger(__name__)

_NODEPAT = re.compile(r'(\d+):\[(.+)\]')

def main():
    logger.info('start load')
    with open('list_xgb_model.pkl', 'rb') as f:
        list_model = pickle.load(f)

    from train_feature_1 import LIST_TRAIN_COL
    feature_column = LIST_TRAIN_COL
    logger.info('num model: %s'%len(list_model))
    model = list_model[-2]
    booster = model.booster()
    map_node = defaultdict(int)
    #print(booster.get_dump()[0])
    for tree in booster.get_dump():
        nodes = [node.split('<')[0] for _, node in _NODEPAT.findall(tree)]
        for i in nodes:
            for j in nodes:
                if i != j and i > j:
                    map_node['%s-%s'%(i, j)] += 1

    df = pandas.Series(dict(map_node)).sort_values(ascending=False)
    df.to_csv('cross_term.csv')
    print(df.head(10))
    """
    list_group = []
    for f1, f2 in [ele.split('-') for ele in df[df > 800].index.values]:
        flg = False
        for g in list_group:
            if f1 in g:
                g.append(f2)
                flg = True
            elif f2 in g:
                g.append(f1)
                flg = True

        if not flg:
            list_group.append([f1, f2])

    import pdb;pdb.set_trace()
    """
if __name__ == '__main__':
    main()
