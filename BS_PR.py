__author__ = 'zhangye'
#this program uses bootstrap to obtain confidence interval for each feature weight
import matplotlib.pyplot as plt
import cPickle as pickle
import predictPR
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import grid_search
import json
import matplotlib.pyplot as plt
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
        print('iteration',i)
    CI_hash_pos = {}
    CI_hash_neg = {}
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
        plt.savefig("BS_PR/"+vectorizer.get_feature_names()[sort_p_lower[i][2]]+"_L2.png")
        plt.clf()

    #draw top features for negative instances
    for i in range(5):
        values = w_lists[:,sort_p_upper[i][2]]
        plt.hist(values,bins=20)
        plt.title(vectorizer.get_feature_names()[sort_p_upper[i][2]])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("BS_PR/"+vectorizer.get_feature_names()[sort_p_upper[i][2]]+"_L2.png")
        plt.clf()
    plot_Features(sort_p_lower,sort_p_upper,X,y,vectorizer)
    '''
    for x in range(50):
        index = sort_p_lower[x][2]
        CI_hash_pos[vectorizer.get_feature_names()[index]]=w_lists[:,index]
    for x in range(50):
        index = sort_p_upper[x][2]
        CI_hash_neg[vectorizer.get_feature_names()[index]]=w_lists[:,index].tolist()

    with open('BS_PR/CI_Hash.json_pos','w') as f:
        json.dump(CI_hash_pos,f)

    with open('/BS_PR/CI_Hash.json_neg','w') as f:
        json.dump(CI_hash_neg,f)
    '''

#plot number of times that top features appearing in positive and negative instances
def plot_Features(sort_p_lower,sort_p_upper,X,y,vectorizer,n=10):
    fig1 = plt.figure()
    for i in range(n):
       feature_Ind = sort_p_lower[i][2]
       ind_pos = np.nonzero(y)
       ind_neg = np.nonzero(y==0)
       sum_pos = np.sum(X[ind_pos,feature_Ind].toarray())
       sum_neg = np.sum(X[ind_neg,feature_Ind].toarray())
       fig1.scatter(sum_pos,sum_neg)
       fig1.annotate(vectorizer.get_feature_names()[feature_Ind],(sum_pos,sum_neg))
    fig1.set_xlabel("number of times in positive instances")
    fig1.set_ylabel("number of times in negative instances")
    fig1.set_title("top features for predicting positive instances")
    fig1.show()
    fig1.savefig("BS_PR/top_features_POS")

    fig2 = plt.figure()
    for i in range(n):
       feature_Ind = sort_p_upper[i][2]
       ind_pos = np.nonzero(y)
       ind_neg = np.nonzero(y==0)
       sum_pos = np.sum(X[ind_pos,feature_Ind].toarray())
       sum_neg = np.sum(X[ind_neg,feature_Ind].toarray())
       fig2.scatter(sum_pos,sum_neg)
       fig2.annotate(vectorizer.get_feature_names()[feature_Ind],(sum_pos,sum_neg))
    fig2.set_xlabel("number of times in positive instances")
    fig2.set_ylabel("number of times in negative instances")
    fig2.set_title("top features for predicting negative instances")
    fig2.show()
    fig2.savefig("BS_PR/top_features_Neg")

def bootstrap_indexes(data, n_samples=1000):
    """
Given data points data, where axis 0 is considered to delineate points, return
an array where each row is a set of bootstrap indexes. This can be used as a list
of bootstrap indexes as well.
    """
    return np.random.randint(data.shape[0],size=(n_samples,data.shape[0]) )

def texify_most_informative_features(sort_p_lower,sort_p_upper,n=50):
    out_str = [
        r'''\begin{table}
            \caption{top 50 features for press release positive prediction}
            \begin{tabular}{l c|l c}

        '''
    ]
    out_str.append(r"\multicolumn{2}{c}{\emph{negative}} & \multicolumn{2}{c}{\emph{positive}} \\")
    for i in range(n):
        out_str.append("%.3f & %s & %.3f & %s \\\\" % (sort_p_upper[i][0], sort_p_upper[i][1],sort_p_lower[i][0], sort_p_lower[i][1]))

    out_str.append(r"\end{tabular}")
    out_str.append(r"\end{table}")

    feature_str = "\n".join(out_str)

    print "\n"
    print feature_str



if __name__ == "__main__":
    con_Interval()
