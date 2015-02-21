__author__ = 'zhangye'
import predictPR
import chambers_analysis
import numpy as np
from sklearn import cross_validation
import itertools
from scipy import vstack
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from scipy.sparse import coo_matrix,vstack,csr_matrix
import scipy.sparse
from sklearn import preprocessing
from sklearn import linear_model
import time
from sklearn.cross_validation import KFold
import pickle
def predict():
    X, y, vectorizer,blocks = get_X_y_block()
    #build a hash table to store blocks
    #key is the block
    #value is the set of document index
    #block_hash = pickle.load(open('save.p',"rb"))
    start_time = time.time()
    block_hash = {}
    for i in range(len(blocks)):
        if(blocks[i] in block_hash):
            block_hash[blocks[i]].append(i)
        else:
            temp = [i]
            block_hash[blocks[i]] = temp
    block_set = block_hash.keys()
    block_data = {}

    #build data set for ranking SVM for each block
    for key in block_hash.keys():
        index = block_hash[key]
        num_document = len(block_hash[key])
        num_pairs = num_document
        comb = itertools.combinations(range(num_document), 2)
        current_Xp = scipy.sparse.csr_matrix((num_pairs,X.shape[1]))
        diff = []
        yp = []
        k = 0
        for (i, j) in comb:
            index_i = index[i]
            index_j = index[j]
            if y[index_i] == y[index_j]:
                # skip if same target
                continue
            current_Xp[k] = X[index_i]-X[index_j]
            diff.append(y[index_i] - y[index_j])
            yp.append(np.sign(diff[-1]))
            if yp[-1] != (-1) ** k:
                yp[-1] *= -1
                current_Xp[k] *= -1
                diff[-1] *= -1
            k += 1
        block_data[key] = (current_Xp,yp)
    elapsed_time = time.time() - start_time
    print("time for converting original dataset into ranking SVM dataset is: ", elapsed_time)

    '''
    Xp = np.zeros((num_pairs,X.shape[1]))
    comb = itertools.combinations(range(X.shape[0]), 2)
    k = 0
    print('convert each pair in one group into one instance')
    original_Index = []
    for (i, j) in comb:
        if y[i] == y[j] \
            or blocks[i] != blocks[j]:
            # skip if same target or different group
            continue
        #Xp = vstack([Xp,(X[i] - X[j]).tocoo()])
        Xp[k] = (X[i] - X[j]).toarray()
        diff.append(y[i] - y[j])
        yp.append(np.sign(diff[-1]))
        original_Index.append((i,j))
        k = k + 1

    # output balanced classes
    for i in range(Xp.shape[0]):
        if yp[i] != (-1) ** i:
            yp[i] *= -1
            Xp[i] *= -1
            diff[i] *= -1

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
    original_Index = []
    for (i, j) in comb:
        if y[i] == y[j] \
            or blocks[i] != blocks[j]:
            # skip if same target or different group
            continue
        #Xp = vstack([Xp,(X[i] - X[j]).tocoo()])
        Xp[k] = (X[i] - X[j]).toarray()
        diff.append(y[i] - y[j])
        yp.append(np.sign(diff[-1]))
        original_Index.append((i,j))
        k = k + 1

    # output balanced classes
    for i in range(Xp.shape[0]):
        if yp[i] != (-1) ** i:
            yp[i] *= -1
            Xp[i] *= -1
            diff[i] *= -1

    yp, diff = map(np.array, (yp, diff))
    '''
    #parameters = {"C":[10, 1, .1, .01, .001,.0001]}
    parameters = [10,1,.1,.01,0.001,0.0001]
    best_auc = 0
    best_C = 0
    best_estimator = linear_model.SGDClassifier()
    print('preprocessing data')
    #Xp = csr_matrix(Xp)
    #Xp = preprocessing.scale(Xp,with_mean=False)

    for p in parameters:
        cv = KFold(len(block_set), n_folds=5)
        auc_for_p = []
        for train, test in cv:
            train_blocks = block_set[train]
            test_blocks = block_set[test]
            Xp = csr_matrix(block_data[train_blocks[0]][0])
            yp = block_data[train_blocks[0]][1]
            for i in range(1,len(train_blocks)):
                vstack(Xp,block_data[train_blocks[i]][0])
                yp.append(block_data[train_blocks[i]][1])
            Xp = preprocessing.scale(Xp,with_mean=False)
            svmModel = linear_model.SGDClassifier(alpha = p,class_weight='auto')
            print('train')
            svmModel.fit(Xp, yp)
            test_index = [j for j in block_hash[i] for i in test_blocks]
            test_data = X[test_index]
            predict = svmModel.decision_function(test_data)
            rank_predict = sorted(zip(list(predict), y[ori_indice]))
            cur_auc = roc_auc_score(y[test_index],predict)
            auc_for_p.append(cur_auc)
        average = sum(auc_for_p)/len(auc_for_p)
        texify_most_informative_features(vectorizer,svmModel,caption="",n=50)
        if(average>best_auc):
            best_auc = average
            best_C = p
            best_estimator = svmModel
    print("best C is: ",best_C)
    print("best auc is:" ,best_auc)

    texify_most_informative_features(vectorizer,best_estimator,"",n=50)


    '''
    for i in range(len(unique_blocks)):
        tau, _ = stats.kendalltau(X_test[b_test == unique_blocks[i]].tocsr().dot(coef), y_test[b_test == unique_blocks[i]])
        print('Kendall correlation coefficient for block %s: %.5f' % (i, tau))
    '''

def texify_most_informative_features(vectorizer, clf, caption, n=50):
    ###
    # note that in the multi-class case, clf.coef_ will
    # have k weight vectors, which I believe are one per
    # each class (i.e., each is a classifier discriminating
    # one class versus the rest).
    #c_f = sorted(zip(clf.coef_[2], vectorizer.get_feature_names()))
    coef = clf.coef_[0]
    coef = coef/ np.linalg.norm(coef)
    c_f = sorted(zip(coef, vectorizer.get_feature_names()))
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