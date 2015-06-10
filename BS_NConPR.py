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
import cPickle
import BS_PR
import math
def con_Interval():
    X, y, vectorizer = get_X_y()
    #BS_PR.sort_ratio(X,y,vectorizer)
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
        print('iteration',i)

    mean = np.mean(w_lists,axis=0)
    std = np.std(w_lists,axis=0)
    p_lower = mean - (1.96)*std
    p_upper = mean + (1.96)*std
    sort_p_lower = sorted(zip(p_lower.tolist(),vectorizer.get_feature_names(),range(len(mean))),reverse=True)
    sort_p_upper = sorted(zip(p_upper.tolist(),vectorizer.get_feature_names(),range(len(mean))))
    save_dict = {}
    save_dict["w_list"] = w_lists
    save_dict["sort_p_lower"] = sort_p_lower
    save_dict["sort_p_upper"] = sort_p_upper
    save_dict["mean"] = list(mean)
    dict_file = open("BS_NConPR/coefficient.pkl","wb")
    cPickle.dump(save_dict,dict_file,cPickle.HIGHEST_PROTOCOL)
    dict_file.close()
    #set break point here
    texify_most_informative_features(sort_p_lower,sort_p_upper)

    #draw top features for positive instances
    for i in range(5):
        values = w_lists[:,sort_p_lower[i][2]]
        plt.hist(values,bins=20)
        plt.title(vectorizer.get_feature_names()[sort_p_lower[i][2]])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("BS_NConPR/"+vectorizer.get_feature_names()[sort_p_lower[i][2]]+"_L2.png")
        plt.clf()

    #draw top features for negative instances
    for i in range(5):
        plt.clf()
        values = w_lists[:,sort_p_upper[i][2]]
        plt.hist(values,bins=20)
        plt.title(vectorizer.get_feature_names()[sort_p_upper[i][2]])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("BS_NConPR/"+vectorizer.get_feature_names()[sort_p_upper[i][2]]+"_L2.png")
        plt.clf()
    plot_Features(sort_p_lower,sort_p_upper,X,y,vectorizer)

#plot number of times that top features appearing in positive and negative instances
def plot_Features(sort_p_lower,sort_p_upper,X,y,vectorizer,n=5):
    for i in range(n):
       feature_Ind = sort_p_lower[i][2]
       ind_pos = np.nonzero(y)
       ind_neg = np.nonzero(y==0)
       sum_pos = np.sum(X[ind_pos,feature_Ind].toarray())
       sum_neg = np.sum(X[ind_neg,feature_Ind].toarray())
       a = plt.scatter(sum_pos,sum_neg,c='blue')
       plt.annotate(vectorizer.get_feature_names()[feature_Ind],(sum_pos,sum_neg))
    plt.xlabel("number of times in positive instances")
    plt.ylabel("number of times in negative instances")
    plt.title("top features for news coverage prediction")

    for i in range(n):
       feature_Ind = sort_p_upper[i][2]
       ind_pos = np.nonzero(y)
       ind_neg = np.nonzero(y==0)
       sum_pos = np.sum(X[ind_pos,feature_Ind].toarray())
       sum_neg = np.sum(X[ind_neg,feature_Ind].toarray())
       b = plt.scatter(sum_pos,sum_neg,c='red')
       plt.annotate(vectorizer.get_feature_names()[feature_Ind],(sum_pos,sum_neg))
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    min_value = min([xmax,ymax])
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.plot(range(int(min_value)),range(int(min_value)),0.01,'-')
    plt.legend((a,b),('positive feature','negative feature'),scatterpoints=1,loc=4)
    #plt.show()
    plt.savefig("BS_NConPR/top_features_NC")


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
    con_Interval()
if __name__ == "main":
    main()