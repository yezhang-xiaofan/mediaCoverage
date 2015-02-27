__author__ = 'zhangye'
import predictPR
import chambers_analysis
import numpy as np
from sklearn import cross_validation
import itertools
from scipy.sparse import vstack
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV
from scipy.sparse import coo_matrix,csr_matrix,lil_matrix
from sklearn import preprocessing
from sklearn import linear_model
import time
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
def predict():
    X, y, vectorizer,blocks = get_X_y_block()
    #build a hash table to store blocks
    #key is the block
    #value is the set of document index
    start_time = time.time()
    block_hash = {}
    for i in range(len(blocks)):
        if(blocks[i] in block_hash):
            block_hash[blocks[i]].append(i)
        else:
            temp = [i]
            block_hash[blocks[i]] = temp

    block_data = {}

    #build data set for ranking SVM for each block
    block_set = []
    single_block = []
    for key in block_hash.keys():
        if(len(block_hash[key])<=1):
            single_block.append(key)
            continue
        block_set.append(key)
        index = block_hash[key]
        num_document = len(block_hash[key])
        num_pair = num_document - 1
        comb = itertools.combinations(range(num_document), 2)
        index_first = []
        index_second = []
        for (i, j) in comb:
            index_i = index[i]
            index_j = index[j]
            if y[index_i] == y[index_j]:
                # skip if same target
                continue
            index_first.append(index_i)
            index_second.append(index_j)
            #index_first.append(index_j)
            #index_second.append(index_i)
        current_Xp = X[index_first] - X[index_second]
        yp = np.sign(y[index_first] - y[index_second])
        block_data[key] = (current_Xp,yp)


        for i in range(num_pair):
            if yp[i] != (-1) ** i:
                yp[i] *= -1
                current_Xp[i] *= -1
        block_data[key] = (current_Xp,yp)


    block_set = np.array(block_set)
    elapsed_time = time.time() - start_time
    single_doc_index = [block_hash[i][0] for i in single_block]  #index for articles without matched samples
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

     output balanced classes
    for i in range(Xp.shape[0]):
        if yp[i] != (-1) ** i:
            yp[i] *= -1
            Xp[i] *= -1
            diff[i] *= -1

    yp, diff = map(np.array, (yp, diff))
    '''
    #parameters = {"C":[10, 1, .1, .01, .001,.0001]}
    parameters = [500,1000,100,10,1,.1,.01,0.001,0.0001]
    best_auc = 0
    best_C = 0
    best_estimator = linear_model.SGDClassifier()
    #Xp = csr_matrix(Xp)
    #Xp = preprocessing.scale(Xp,with_mean=False)

    for p in parameters:
        cv = KFold(len(block_set), n_folds=5,shuffle=True)
        auc_for_p = []
        for train, test in cv:
            train_blocks = block_set[train]
            test_blocks = block_set[test]
            num_row = 0
            for block in train_blocks:
                num_row += block_data[block][0].shape[0]
            Xp = np.empty((num_row,X.shape[1]))
            yp = np.empty(num_row)
            k = 0
            for i in range(len(train_blocks)):
                temp_num = block_data[train_blocks[i]][0].shape[0]
                Xp[k:k+temp_num,:] = block_data[train_blocks[i]][0].toarray()
                yp[k:k+temp_num] = block_data[train_blocks[i]][1]
                k += temp_num
            min_max_scaler = preprocessing.MinMaxScaler()
            Xp = min_max_scaler.fit_transform(Xp)
            #Xp = preprocessing.scale(Xp)
            svmModel = linear_model.SGDClassifier(alpha = p,class_weight='auto')
            print('train')
            svmModel.fit(Xp, yp)
            test_index = [j for i in test_blocks for j in block_hash[i]]
            test_data = X[test_index].tocsr()
            test_data = vstack([test_data,X[single_doc_index].tocsr()])
            test_data = min_max_scaler.transform(test_data.toarray())
            test_label = np.hstack((y[test_index],y[single_doc_index]))
            predict = svmModel.decision_function(test_data)
            rank_predict = sorted(zip(list(predict), test_label))
            cur_auc = roc_auc_score(test_label,predict)
            auc_for_p.append(cur_auc)
        average = sum(auc_for_p)/len(auc_for_p)
        print("current average auc_score is: ",p, average)
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