__author__ = 'zhangye'
#this program uses L2 logistic regression to predict new coverage conditioned on press release
#on Reuters dataset
from factiva_model import process_file
from BS_PR import bootstrap_indexes
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.feature_extraction.text import CountVectorizer
import math
import matplotlib.pyplot as plt
from BS_PR import texify_most_informative_features
from sklearn.linear_model import SGDClassifier
import cPickle
def con_Interval():
    X, y, vectorizer = get_X_y()
    n_samples = 1000
    bs_indexes = bootstrap_indexes(X,n_samples)
    w_lists = np.zeros((n_samples,X.shape[1]))
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
    kf = cross_validation.StratifiedKFold(y,n_folds=5,shuffle=True)
    parameters = {"C":[100,10,1.0,.1, .01, .001,0.0001]}
    clf0 = grid_search.GridSearchCV(lr, parameters,scoring='roc_auc',cv=kf)
    clf0.fit(X,y)
    best_C = clf0.best_params_['C']
    print "best AUC score is: " + str(clf0.best_score_)
    for i in range(n_samples):
        train_X = X[bs_indexes[i]]
        train_Y = y[bs_indexes[i]]
        #lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto',C=best_C)
        #lr.fit(train_X,train_Y)
        clf = SGDClassifier(loss="log",alpha=1.0/best_C,n_iter=np.ceil(10**6/train_X.shape[0]),class_weight="auto").fit(train_X, train_Y)
        w = clf.coef_
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
    save_dict = {}
    save_dict["w_list"] = w_lists
    save_dict["sort_p_lower"] = sort_p_lower
    save_dict["sort_p_upper"] = sort_p_upper
    dict_file = open("reuters/coefficient.pkl","wb")
    cPickle.dump(save_dict,dict_file,cPickle.HIGHEST_PROTOCOL)
    dict_file.close()
    texify_most_informative_features(sort_p_lower,sort_p_upper)
    #draw top features for positive instances
    plot_index = [1,2,4,8]
    plt.figure(1)

def get_X_y():
    X,y,interest = process_file("reuters/all_reuters_article_info.csv","reuters/all_reuters_matched_articles_filtered.csv")
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english",
                                    min_df=1,
                                    token_pattern=r"(?u)95% confidence interval|95% CI|95% ci|[a-zA-Z0-9_*\-][a-zA-Z0-9_/*\-]+",
                                    binary=False, max_features=50000)
    X = vectorizer.fit_transform(X)
    return X,np.array(y),vectorizer

def main():
    con_Interval()

if __name__ == "__main__":
    main()