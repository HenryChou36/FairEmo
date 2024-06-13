import numpy as np

def PCC(x, y):
    ''' Pearson correlation coefficient'''
    v_x = x - np.mean(x, axis=0)
    v_y = y - np.mean(y, axis=0)
    pcc = np.sum(v_x * v_y, axis=0) / \
        (np.sqrt(np.sum(v_x ** 2, axis=0) * np.sum(v_y ** 2, axis=0)  + 1e-8))
    return pcc

def CCC(x, y):
    ''' Concordance correlation coefficient'''
    pcc = PCC(x, y)
    ccc = 2 * pcc * np.std(x, axis=0) * np.std(y, axis=0) / \
        (np.var(x, axis=0) + np.var(y, axis=0) + \
            ((np.mean(x, axis=0) - np.mean(y, axis=0))**2) + 1e-8)
    return ccc

def SPC(x, y):
    ''' Spearman rank correlation coefficient'''
    x_temp = np.argsort(x, axis=0)
    x_ranks = np.empty_like(x_temp).astype('f')
    if x_temp.ndim == 1:
        x_ranks[x_temp] = np.arange(x_temp.shape[0], dtype='f')
    else:
        order = np.linspace(start=np.zeros(x_temp.shape[1]),
                            stop=np.ones(x_temp.shape[1]) * (x_temp.shape[0] - 1),
                            num=x_temp.shape[0], axis=0, dtype='f')
        np.put_along_axis(x_ranks, x_temp, order, axis=0)
    
    y_temp = np.argsort(y, axis=0)
    y_ranks = np.empty_like(y_temp).astype('f')
    if y_temp.ndim == 1:
        y_ranks[y_temp] =   np.arange(y_temp.shape[0], dtype='f')
    else:
        order = np.linspace(start=np.zeros(x_temp.shape[1]),
                            stop=np.ones(x_temp.shape[1]) * (x_temp.shape[0] - 1),
                            num=x_temp.shape[0], axis=0, dtype='f')
        np.put_along_axis(y_ranks, y_temp, order, axis=0)

    spc = PCC(x_ranks, y_ranks)

    return spc

class Discrimination(object):
    def __init__(self, min_sample=32):
        self.min_sample=min_sample
    def __call__(self, pred, attribute):
        # ensure no nan
        att = attribute[~np.isnan(pred).any(axis=1)]
        pred = pred[~np.isnan(pred).any(axis=1)]

        # select group with enough sample size
        att_val, att_cnt = np.unique(att, return_counts = True)
        att_val = att_val[np.where(att_cnt >= self.min_sample)]
        mean = np.mean(pred, axis=0)
        parity_diff = np.zeros(pred.shape[-1])
        for i, val in enumerate(att_val):
            parity_diff += np.abs(np.mean(pred[np.where(att == val)], axis=0) - mean)

        return 2 * parity_diff / len(att_val)
