__author__ = 'zhangye'

#This program uses bootstrap to predict news coverage conditioned on press release

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import grid_search
print "haha"
from predictNConPR import get_X_y
from BS_PR import bootstrap_indexes
import matplotlib.pyplot as plt
def con_Interval():
    X, y, vectorizer = get_X_y()
    n_samples = 1000
    bs_indexes = bootstrap_indexes(X,n_samples)
    w_lists = np.zeros((n_samples,X.shape[1]))
    lr = LogisticRegression(penalty="l1", fit_intercept=True,class_weight='auto')
    kf = cross_validation.StratifiedKFold(y,n_folds=5,shuffle=True)
    parameters = {"C":[100,10,1.0,.1, .01, .001,0.0001]}
    clf0 = grid_search.GridSearchCV(lr, parameters,scoring='roc_auc',cv=kf)
    clf0.fit(X,y)
    best_C = clf0.best_params_['C']

    for i in range(n_samples):
        train_X = X[bs_indexes[i]]
        train_Y = y[bs_indexes[i]]
        lr = LogisticRegression(penalty="l1", fit_intercept=True,class_weight='auto',C=best_C)
        lr.fit(train_X,train_Y)
        w = lr.coef_
        w_lists[i] = w
        print('iteration',i)

    mean = np.mean(w_lists,axis=0)
    std = np.std(w_lists,axis=0)
    p_lower = mean - (1.96)*std
    p_upper = mean + (1.96)*std
    sort_p_lower = sorted(zip(p_lower.tolist(),vectorizer.get_feature_names(),range(len(mean))),reverse=True)
    sort_p_upper = sorted(zip(p_upper.tolist(),vectorizer.get_feature_names(),range(len(mean))))
    texify_most_informative_features(sort_p_lower,sort_p_upper)

    #draw top features for positive instances
    for i in range(5):
        values = w_lists[:,sort_p_lower[i][2]]
        plt.hist(values,bins=20)
        plt.title(vectorizer.get_feature_names()[sort_p_lower[i][2]])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("BS_NConPR/"+vectorizer.get_feature_names()[sort_p_lower[i][2]]+"_L1.png")
        plt.clf()

    #draw top features for negative instances
    for i in range(5):
        plt.clf()
        values = w_lists[:,sort_p_upper[i][2]]
        plt.hist(values,bins=20)
        plt.title(vectorizer.get_feature_names()[sort_p_upper[i][2]])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("BS_NConPR/"+vectorizer.get_feature_names()[sort_p_upper[i][2]]+"_L1.png")
        plt.clf()

def texify_most_informative_features(sort_p_lower,sort_p_upper,n=50):
    out_str = [
        r'''\begin{table}
            \caption{top 50 features for news coverage prediction conditioned on press release}
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