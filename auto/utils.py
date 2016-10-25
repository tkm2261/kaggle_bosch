import numpy
import pandas
#from numba.decorators import jit


def mcc(y_true, y_pred):
    n = y_true.shape[0]
    true_pos = sum(1. for i in range(n) if y_true[i] == 1 and y_pred[i] == 1)
    true_neg = sum(1. for i in range(n) if y_true[i] == 0 and y_pred[i] == 0)
    false_pos = sum(1. for i in range(n) if y_true[i] == 0 and y_pred[i] == 1)
    false_neg = sum(1. for i in range(n) if y_true[i] == 1 and y_pred[i] == 0)

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
        if score > max_score:
            max_score = score
    return max_score


def mcc_scoring2(y_pred_prb, y):
    list_thresh = numpy.arange(1, 100) / 100
    max_score = -1
    idx = None
    for thresh in list_thresh:
        y_pred = numpy.where(y_pred_prb >= thresh, 1, 0)
        score = mcc(y, y_pred)
        if score > max_score:
            max_score = score
            idx = thresh
    return idx, max_score


def mcc_optimize(y_prob, y_true):

    df = pandas.DataFrame()
    df['y_prob'] = y_prob
    df['y_true'] = y_true
    df = df.sort_values('y_prob')
    y_prob_sort = df['y_prob'].values
    y_true_sort = df['y_true'].values
    n = y_true.shape[0]
    nump = y_true.sum()
    numn = n - nump

    tn_v = numpy.cumsum(y_true_sort == 0, dtype=numpy.int)
    fp_v = numpy.cumsum(y_true_sort == 1, dtype=numpy.int)
    fn_v = numn - tn_v
    tp_v = nump - fp_v
    s = (tp_v + fn_v) / n
    p = (tp_v + fp_v) / n
    sup_v = tp_v / n - s * p
    inf_v = numpy.sqrt(p * s * (1 - p) * (1 - s))
    mcc_v = sup_v / inf_v
    mcc_v[numpy.isinf(mcc_v)] = -1
    mcc_v[mcc_v != mcc_v] = -1

    df = pandas.DataFrame()
    df['mcc'] = mcc_v
    df['pred'] = y_prob_sort
    df = df.sort_values(by='mcc', ascending=False).reset_index(drop=True)

    best_mcc = df.ix[0, 'mcc']
    best_proba = df.ix[0, 'pred']

    return best_proba, best_mcc


def evalmcc_xgb_min(preds, dtrain):
    labels = dtrain.get_label()
    best_proba, best_mcc = mcc_optimize(preds, labels)
    return 'MCC', - best_mcc


if __name__ == '__main__':

    y_prob = numpy.random.random(10000)
    y_true = numpy.where(numpy.random.random(10000) > 0.5, 1, 0)
    print(mcc_optimize(y_prob, y_true))
    print(mcc_scoring2(y_prob, y_true))
