__author__ = 'zhangye'
import csv
import pdb
import os
import chambers_analysis
import sklearn
from sklearn.metrics import auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
import predictPR
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif,SelectKBest

import pylab
def predict_NC():
    #feature selection
    X, y, vectorizer = get_X_y()
    #selector = SelectKBest(f_classif,10000)
    selector = SelectPercentile(f_classif,percentile=100)
    selector.fit(X,y)
    best_indices = selector.get_support(indices=True)
    best_features = np.array(vectorizer.get_feature_names())[best_indices]
    X = selector.transform(X)

    #use cross validation to choose the best parameter
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
    kf = StratifiedKFold(y,n_folds=5,shuffle=True)
    parameters = {"C":[1.0,.1, .01, .001,0.0001]}
    clf0 = GridSearchCV(lr, parameters,scoring='roc_auc',cv=kf)
    print "fitting model..."
    clf0.fit(X,y)
    print "best auc score is: " ,clf0.best_score_
    print "done."

    #cross validation on the best parameter
    #get precision recall accuracy auc_score
    fs, aucs,prec,rec = [],[],[],[]
    fold = 0
    complete_X = X.tocsr()
    clf = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto',C=clf0.best_estimator_.C)
    for train, test in kf:
        clf.fit(complete_X[train,:].tocoo(), y[train])
        probs = clf.predict_proba(complete_X[test,:])[:,1]
        average_precision_score(y[test],probs)
        precision,recall,threshold = precision_recall_curve(y[test],probs)

        accuracy = clf.score(complete_X[test,:], y[test])

        predLabel = clf.predict(X[test,:])
        rec.append(recall_score(y[test],predLabel))
        prec.append(precision_score(y[test],predLabel))
        #aucs.append(sklearn.metrics.roc_auc_score(y[test], probs))
        cur_auc = auc_score(y[test], probs)
        aucs.append(cur_auc)
        #preds = clf.predict(complete_X[test])
        #fs.append(f1_score(y[test], preds))
        '''
        if fold == 0:
            plt.clf()
            plt.plot(precision,recall,label='Precision-Recall curve for news coverage prediction')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0,1.05])
            plt.xlim([0.0,1.0])
            plt.title('Precision-Recall curve for news coverage prediction with vocabulary size %d' %len(best_features))
            plt.show()
        fold += 1
        '''

        if fold == 0:
            fpr, tpr, thresholds = roc_curve(y[test], probs)
            pylab.clf()
            fout = "NC/roc"

            pylab.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % cur_auc)
            pylab.plot([0,1], [0,1], 'k--')
            pylab.xlim((-0.025,1.025))
            pylab.ylim((-0.025,1.025))
            pylab.xlabel("false positive rate")
            pylab.ylabel("true positive rate")
            pylab.title("ROC curve for news coverage prediction(area = %0.2f)" % cur_auc)
            pylab.tight_layout()
            pylab.savefig(fout)
        fold += 1

    #print "average auc: %s" % (sum(aucs)/float(len(aucs)))
    #print "average fs: %s" % (sum(fs)/float(len(fs)))
    print "average recall: %s" % (sum(rec)/float(len(rec)))
    print "average precision: %s" % (sum(prec)/float(len(prec)))
    #print "ABOUT TO RETURN"
    #pdb.set_trace()
    texify_most_informative_features(best_features,vectorizer, clf0)
    return clf0

def get_X_y():
    '''
    Get X and y for the task of predicting whether a given
    article will get a news article. (Recall that all articles
    in this corpus will receive press releases!)
    '''
    articles = load_articles()
    matchSample = predictPR.load_matched_samples()
    article_texts = [process_article(article) for article in articles if article["has_news_article"]==1]
    matchSampe_texts = [chambers_analysis.process_article(article) for article in matchSample]
    all_texts = article_texts+matchSampe_texts
    vectorizer = get_vectorizer(all_texts)
    X = vectorizer.transform(all_texts)
    #transformer = TfidfTransformer()
    #X = transformer.fit_transform(X)

    y = []
    for article in article_texts:
        y.append(1)
    for match in matchSampe_texts:
        y.append(0)
    y = np.array(y)
    print "read %s articles with news coverage." % len(article_texts)
    print "read %s articles without news coverage. " % len(matchSampe_texts)
    return X, y, vectorizer


def get_vectorizer(article_texts, max_features=50000):
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english",
                                    min_df=2,
                                    token_pattern=r"(?u)[a-zA-Z0-9-_/*][a-zA-Z0-9-_/*]+\b",
                                    binary=False, max_features=max_features)
    vectorizer.fit(article_texts)
    return vectorizer


def load_articles(articles_dir="1. excel files",
                    news_articles_dir="6. News articles"):

    article_files = []
    for file_name in os.listdir(articles_dir):
        if file_name.endswith(".article_info.txt"):
            article_files.append(file_name)
            
    articles = []
    for article_file_name in article_files:
        article = read_in_article(os.path.join(articles_dir, article_file_name))
        # does it have an associated news article?
        # (note that all articles are assumed to
        #  have press releases, by construction)
        # to assess we check if the corresponding file
        # exists. these look like "01-11-002.txt",
        # which matches "01-11-002-1.xls".
        article_identifier = article_file_name.split(".")[0]
        # not entirely sure what's up with the '-1'
        # they append to the files, but this needs to
        # be removed.
        article_identifier = article_identifier + ".txt"
        article_identifier = article_identifier.replace("-1.txt", ".txt")
        has_news_article = os.path.exists(
                                os.path.join(news_articles_dir,
                                article_identifier))

        article["has_news_article"] = 1 if has_news_article else 0
        articles.append(article)
    return articles

def read_in_article(article_path):
    article_dict = {}
    with open(article_path, 'rU') as input_file:
        reader = csv.reader(input_file)
        pmid, title, mesh, authors, abstract, affiliation, journal, volume = reader.next()


    return {"pmid":pmid, "title":title, "mesh":mesh, "authors":authors,
                "abstract":abstract, "affiliation":affiliation}





##########################################################
# Below we define several utility methods.
#
# A fair amount of this is copy-pasta'd from the factiva
# module @TODO re-factor to avoid this duplication.
##########################################################

def process_article(article):
    def prepend_to_words(text, prepend_str="TI-", split_on=" "):
        return " ".join(
                [prepend_str + text for text in text.split(split_on) if not
                        text.strip() in ("missing", "", "&") and not
                        text.strip() in sklearn.feature_extraction.text.ENGLISH_STOP_WORDS])

    def clean_up_mesh(mesh):

        mesh_tokens = mesh.split(" ")
        clean_tokens = []
        remove_these = ["&", "(", ")"]
        for t in mesh_tokens:
            for to_remove in remove_these:
                t = t.replace(to_remove, "")

            if t.startswith("*"):
                t = t[1:]

            if len(t) > 0:
                clean_tokens.append(t)
        return " ".join(clean_tokens)


    #pdb.set_trace()
    all_features = [prepend_to_words(article["title"], prepend_str="TI-"),
                    article["abstract"],
                    #prepend_to_words(article["affiliation"], prepend_str="AF-"),
                    prepend_to_words(article["mesh"].replace(" ", "-"),
                                prepend_str="MH-", split_on="\n")]


    #all_features = [article["title"], article["abstract"],
    #                prepend_to_words(article["mesh"], prepend_str="MH-")]

    return " ".join(all_features)

def _get_ranked_features(clf, vectorizer, n=50):
    c_f = sorted(zip(clf.best_estimator_.raw_coef_[0], vectorizer.get_feature_names()))

    weights_d = dict(zip(vectorizer.get_feature_names(), clf.best_estimator_.raw_coef_[0]))
    return c_f, weights_d


def texify_most_informative_features(best_features,vectorizer,clf, caption='', n=50):
    ###
    # note that in the multi-class case, clf.coef_ will
    # have k weight vectors, which I believe are one per
    # each class (i.e., each is a classifier discriminating
    # one class versus the rest).
    #c_f = sorted(zip(clf.coef_[2], vectorizer.get_feature_names()))
    c_f = sorted(zip(clf.best_estimator_.raw_coef_[0], best_features))
    if n == 0:
        n = len(c_f)/2

    top = zip(c_f[:n], c_f[:-(n+1):-1])
    print
    print "%d most informative features:" % (n, )
    out_str = [
        r'''\begin{table}
            \caption{top fifty features and associated weight for news coverage prediction}
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

def show_most_informative_features(vectorizer, clf, n=50):
    ###
    # note that in the multi-class case, clf.coef_ will
    # have k weight vectors, which I believe are one per
    # each class (i.e., each is a classifier discriminating
    # one class versus the rest).
    #c_f = sorted(zip(clf.coef_[2], vectorizer.get_feature_names()))
    c_f = sorted(zip(clf.best_estimator_.raw_coef_[0], vectorizer.get_feature_names()))
    if n == 0:
        n = len(c_f)/2

    top = zip(c_f[:n], c_f[:-(n+1):-1])
    print
    print "%d most informative features:" % (n, )
    out_str = []
    for (c1, f1), (c2, f2) in top:
        out_str.append("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))
    feature_str = "\n".join(out_str)
    print feature_str

def main():
    predict_NC()
main()
