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
import cPickle
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pickle
def con_Interval():
    X, y, vectorizer = predictPR.get_X_y()
    #sort_ratio(X,y,vectorizer)
    n_samples = 1000
    bs_indexes = bootstrap_indexes(X,n_samples)
    w_lists = np.zeros((n_samples,X.shape[1]))
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
    kf = cross_validation.StratifiedKFold(y,n_folds=5,shuffle=True)
    parameters = {"C":[100,10,1.0,.1, .01, .001,0.0001]}
    clf0 = grid_search.GridSearchCV(lr, parameters,scoring='roc_auc',cv=kf)
    clf0.fit(X,y)
    print "best auc score is: " + str(clf0.best_score_)
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
    #sort_p_* is list of tuples
    #the first is lower/upper bound of CI
    #the second is name of feature
    #the third is the index in the original feature vector
    sort_p_lower = sorted(zip(p_lower.tolist(),vectorizer.get_feature_names(),range(len(mean))),reverse=True)
    sort_p_upper = sorted(zip(p_upper.tolist(),vectorizer.get_feature_names(),range(len(mean))))
    save_dict = {}
    save_dict["w_list"] = w_lists
    save_dict["sort_p_lower"] = sort_p_lower
    save_dict["sort_p_upper"] = sort_p_upper
    save_dict["mean"] = list(mean)
    dict_file = open("BS_PR/coefficient.pkl","wb")
    cPickle.dump(save_dict,dict_file,cPickle.HIGHEST_PROTOCOL)
    dict_file.close()
    texify_most_informative_features(sort_p_lower,sort_p_upper)
    #draw top features for positive instances
    plot_index = [1,2,4,8]
    plt.figure(1)
    for i in range(len(plot_index)):
        values = w_lists[:,sort_p_lower[plot_index[i]][2]]
        plt.subplot(2,2,i+1)
        values = values/1000.0
        sns.kdeplot(values)
        plt.title(vectorizer.get_feature_names()[sort_p_lower[i][2]])
        if(i==2):
            plt.xlabel("Coefficient Value")
            plt.ylabel("Density")
        #plt.savefig("BS_PR/"+vectorizer.get_feature_names()[sort_p_lower[i][2]]+"_L2.png")
        #plt.clf()

    plt.savefig("BS_PR/"+"top_fea_pos+"+"_L2.png")


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
def plot_Features(sort_p_lower,sort_p_upper,X,y,vectorizer,n=5):
    #change the range interested here
    plot_index = [1,2,4,8]
    plt.figure(1)
    for i in range(len(plot_index)):
       plt.plot(2,2,i+1)
       feature_Ind = sort_p_lower[plot_index[i]][2]
       ind_pos = np.nonzero(y)
       ind_neg = np.nonzero(y==0)
       sum_pos = np.sum(X[ind_pos,feature_Ind].toarray())
       sum_neg = np.sum(X[ind_neg,feature_Ind].toarray())
       a = plt.scatter(sum_pos,sum_neg,c='blue')
       plt.annotate(vectorizer.get_feature_names()[feature_Ind],(sum_pos,sum_neg))
       plt.xlabel("times in positive instances")
       plt.ylabel("times in negative instances")
    plt.title("top features for press release prediction")
    plt.savefig("BS_PR/top_features_pos_PR")

    plt.figure(2)
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
    plt.legend((a,b),('positive feature','negative feature'),scatterpoints=1,loc=2)
    plt.savefig("BS_PR/top_features_neg_PR")
    plt.close()

def sort_ratio(X,y,vectorizer,n=50):
    ind_pos = np.nonzero(y)
    ind_neg = np.nonzero(y==0)
    sum_pos = np.sum(X[ind_pos].toarray(),axis=0).astype('float')
    sum_neg = np.sum(X[ind_neg].toarray(),axis=0).astype('float')
    ratio1 = np.divide(sum_pos,sum_neg)
    ratio2 = np.divide(sum_neg,sum_pos)
    ratio1_sort = sorted(zip(ratio1,vectorizer.get_feature_names()),reverse=True)
    ratio2_sort = sorted(zip(ratio2,vectorizer.get_feature_names()),reverse=True)
    ratio1_sort = [i for i in ratio1_sort if i[0]!=float('Inf')]
    ratio2_sort = [i for i in ratio2_sort if i[0]!=float('Inf')]
    texify_most_informative_features(ratio1_sort,ratio2_sort,n=100)

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
            \caption{top 100 features for press release positive prediction}
            \begin{tabular}{l c|l c}

        '''
    ]
    out_str.append(r"\multicolumn{2}{c}{\emph{negative}} & \multicolumn{2}{c}{\emph{positive}} \\")
    i = 0
    while i<n:
        out_str.append("%.5f & %s & %.5f & %s \\\\" % (sort_p_upper[i][0], sort_p_upper[i][1],sort_p_lower[i][0], sort_p_lower[i][1]))
        i += 1

    out_str.append(r"\end{tabular}")
    out_str.append(r"\end{table}")

    feature_str = "\n".join(out_str)

    print "\n"
    print feature_str



if __name__ == "__main__":
    con_Interval()
