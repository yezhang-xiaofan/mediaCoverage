__author__ = 'zhangye'
import rankSVM
import csv
import os
from os.path import basename
import predictPR
import chambers_analysis
import numpy as np
from sklearn import cross_validation
from scipy import stats
import itertools
from scipy import vstack
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from scipy.sparse import coo_matrix,vstack,csr_matrix
import scipy.sparse
from sklearn import preprocessing
def predict():
    X, y, vectorizer,blocks = get_X_y_block()
    comb = itertools.combinations(range(X.shape[0]), 2)
    Xp, yp, diff = [], [], []
    num_pairs = 0
    for (i, j) in comb:
        if y[i] == y[j] \
            or blocks[i] != blocks[j]:
            # skip if same target or different group
            continue
        num_pairs  = num_pairs + 1

    Xp = np.zeros((num_pairs,X.shape[1]))
    comb = itertools.combinations(range(X.shape[0]), 2)
    k = 0
    print('convert each pair in one group into one instance')
    for (i, j) in comb:
        if y[i] == y[j] \
            or blocks[i] != blocks[j]:
            # skip if same target or different group
            continue
        #Xp = vstack([Xp,(X[i] - X[j]).tocoo()])
        Xp[k] = (X[i] - X[j]).toarray()
        diff.append(y[i] - y[j])
        yp.append(np.sign(diff[-1]))
        k = k + 1

    # output balanced classes

    for i in range(Xp.shape[0]):
        if yp[i] != (-1) ** i:
            yp[i] *= -1
            Xp[i] *= -1
            diff[i] *= -1

    yp, diff = map(np.array, (yp, diff))

    #parameters = {"C":[10, 1, .1, .01, .001,.0001]}
    parameters = [1,.1,.01,.001,0.003,0.005]
    # cross validation
    best_auc = 0
    best_C = 0
    best_estimator = svm.SVC()

    '''
    clf = GridSearchCV(svmModel,parameters,scoring='roc_auc',cv=cv)
    Xp = coo_matrix(Xp)
    print('begin to train')
    clf.fit(Xp,yp)
    '''
    print('preprocessing data')
    Xp = csr_matrix(Xp)
    Xp = preprocessing.scale(Xp,with_mean=False)
    #Xp = Xp.toarray()
    for p in parameters:
        cv = cross_validation.StratifiedShuffleSplit(yp, test_size=.2)
        auc_for_p = []
        for train, test in cv:
            X_train, y_train = Xp[train], yp[train]
            X_test, y_test = Xp[test], yp[test]
            svmModel = svm.SVC(C=p,kernel='linear')
            #clf0 = GridSearchCV(svmModel, parameters,scoring='roc_auc')
            print('train')
            svmModel.fit(X_train.tocoo(), y_train)
            best_estimator = svmModel
            predict = svmModel.decision_function(X_test.tocoo())
            #coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)
            cur_auc = roc_auc_score(y_test,predict)
            auc_for_p.append(cur_auc)
        average = sum(auc_for_p)/len(auc_for_p)
        if(average>best_auc):
            best_auc = average
            best_C = p
            best_estimator = svmModel

    '''
    for i in range(len(unique_blocks)):
        tau, _ = stats.kendalltau(X_test[b_test == unique_blocks[i]].tocsr().dot(coef), y_test[b_test == unique_blocks[i]])
        print('Kendall correlation coefficient for block %s: %.5f' % (i, tau))
    '''

    texify_most_informative_features(vectorizer,best_estimator,caption="",n=50)

def texify_most_informative_features(vectorizer, clf, caption, n=50):
    ###
    # note that in the multi-class case, clf.coef_ will
    # have k weight vectors, which I believe are one per
    # each class (i.e., each is a classifier discriminating
    # one class versus the rest).
    #c_f = sorted(zip(clf.coef_[2], vectorizer.get_feature_names()))
    c_f = sorted(zip(clf.coef_, vectorizer.get_feature_names()))
    if n == 0:
        n = len(c_f)/2

    top = zip(c_f[:n], c_f[:-(n+1):-1])
    print
    print "%d most informative features:" % (n, )
    out_str = [
        r'''\begin{table}
            \begin{tabular}{l c | l c}

        '''
    ]
    out_str.append(r"\multicolumn{2}{c}{\emph{negative}} & \multicolumn{2}{c}{\emph{positive}} \\")
    for (c1, f1), (c2, f2) in top:
        #out_str.append("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))
        out_str.append("%.3f & %s & %.3f & %s \\\\" % (c1, f1, c2, f2))

    #
    out_str.append(r"\end{tabular}")
    out_str.append("%s" % caption)
    out_str.append(r"\end{table}")

    feature_str = "\n".join(out_str)

    print "\n"
    print feature_str
    #return (feature_str, top)

def get_X_y_block():
    '''
    Get X and y for the task of predicting whether a given
    article will get a press release.
    '''
    articles = predictPR.load_articles()
    matchSample = predictPR.load_matched_samples()

    article_texts = [chambers_analysis.process_article(article) for article in articles]
    matchSampe_texts = [chambers_analysis.process_article(article) for article in matchSample]

    all_texts = article_texts+matchSampe_texts[:]
    vectorizer = chambers_analysis.get_vectorizer(all_texts)

    x = vectorizer.transform(all_texts)
    #transformer = TfidfTransformer()
    #X = transformer.fit_transform(X)
    y = []
    blocks = []
    for article in articles:
        y.append(1)
        blocks.append(article["block"])
    for article in matchSample:
        y.append(0)
        blocks.append(article["block"])
    y = np.array(y)
    blocks = np.array(blocks)
    return x, y, vectorizer,blocks

def main():
    predict()

main()