__author__ = 'zhangye'

#This program uses bootstrap to predict news coverage conditioned on press release
import predictNConPR
import BS_PR
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import grid_search

def con_Interval():
    X, y, vectorizer = predictNConPR.get_X_y()
    n_samples = 1000
    bs_indexes = BS_PR.bootstrap_indexes(X,n_samples)
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
        print('iteration',i)

    mean = np.mean(w_lists,axis=0)
    std = np.std(w_lists,axis=0)
    p_lower = mean - (1.96)*std
    p_upper = mean + (1.96)*std
    sort_p_lower = sorted(zip(p_lower.tolist(),vectorizer.get_feature_names()),reverse=True)
    sort_p_upper = sorted(zip(p_upper.tolist(),vectorizer.get_feature_names()))
    texify_most_informative_features(sort_p_lower,sort_p_upper)

def texify_most_informative_features(sort_p_lower,sort_p_upper,n=50):
    out_str = [
        r'''\begin{table}
            \caption{top 50 features for press release prediction conditioned on news coverage}
            \begin{tabular}{l c|l c}
        '''
    ]
    out_str.append(r"\multicolumn{2}{c}{\emph{negative}} & \multicolumn{2}{c}{\emph{positive}} \\")
    for i in range(n):
        out_str.append("%.5f & %s & %.5f & %s \\\\" % (sort_p_upper[i][0], sort_p_upper[i][1],sort_p_lower[i][0], sort_p_lower[i][1]))

    out_str.append(r"\end{tabular}")
    out_str.append(r"\end{table}")

    feature_str = "\n".join(out_str)

    print "\n"
    print feature_str
def main():
    print "haha"
    con_Interval()
main()