__author__ = 'zhangye'
#this program uses bootstrap to obtain confidence interval for each feature weight

import predictPR
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import grid_search
def con_Interval():
    X, y, vectorizer = predictPR.get_X_y()
    n_samples = 1000
    bs_indexes = bootstrap_indexes(X,n_samples)
    w_lists = np.zeros((n_samples,X.shape[1]))
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
    kf = cross_validation.StratifiedKFold(y,n_folds=5,shuffle=True)
    parameters = {"C":[100,10,1.0,.1, .01, .001,0.0001]}
    clf0 = grid_search.GridSearchCV(lr, parameters,scoring='roc_auc',cv=kf)
    clf0.fit(X,y)
    best_C = clf0.best_params_['C']
    for i in range(n_samples):
        train_X = X[bs_indexes[i]]
        train_Y = y[bs_indexes[i]]
        lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto',C=best_C)
        lr.fit(train_X,train_Y)
        w = lr.coef_
        w_lists[i] = w

    mean = np.mean(w_lists,axis=0)
    std = np.std(w_lists,axis=0)
    p = mean - (1.96)*std
    sort_p = sorted(zip(p.tolist(),vectorizer.get_feature_names()))


def bootstrap_indexes(data, n_samples=1000):
    """
Given data points data, where axis 0 is considered to delineate points, return
an array where each row is a set of bootstrap indexes. This can be used as a list
of bootstrap indexes as well.
    """
    return np.random.randint(data.shape[0],size=(n_samples,data.shape[0]) )

def main():
    con_Interval()

main()