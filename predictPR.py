__author__ = 'zhangye'
import csv
import pdb
import os
import chambers_analysis
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

import numpy as np
from numpy.linalg import norm

import pylab
def predict_PR():

    X, y, vectorizer = get_X_y()
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')

    parameters = {"C":[1, .1, .01, .001,.0001]}
    clf0 = GridSearchCV(lr, parameters,scoring='roc_auc')
    print "fitting model..."
    clf0.fit(X,y)
    print "done."

    #print texify_most_informative_features(vectorizer, clf0, "predictive features")

    kf = cross_validation.KFold(X.shape[0], shuffle=True, n_folds=5)
    fs, aucs = [],[]
    fold = 0
    complete_X = X.tocsr()
    for train, test in kf:
        clf = GridSearchCV(lr, parameters,scoring='roc_auc')
        clf.fit(complete_X[train,:].tocoo(), y[train])

        probs = clf.predict_proba(complete_X[test,:])
        #aucs.append(sklearn.metrics.roc_auc_score(y[test], probs))
        cur_auc = roc_auc_score(y[test], probs[:,1])
        aucs.append(cur_auc)
        preds = clf.predict(complete_X[test])
        fs.append(f1_score(y[test], preds))
        if fold == 0:
            fpr, tpr, thresholds = roc_curve(y[test], probs[:,1])
            pylab.clf()
            fout = "roc"

            pylab.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % cur_auc)
            pylab.plot([0,1], [0,1], 'k--')
            pylab.xlim((-0.025,1.025))
            pylab.ylim((-0.025,1.025))
            pylab.xlabel("false positive rate")
            pylab.ylabel("true positive rate")
            pylab.title("ROC curve (area = %0.2f)" % cur_auc)
            pylab.tight_layout()
            pylab.savefig(fout)


        fold += 1

    print "average auc: %s" % (sum(aucs)/float(len(aucs)))
    print "average fs: %s" % (sum(fs)/float(len(fs)))
    #print "ABOUT TO RETURN"
    #pdb.set_trace()
    return clf0


def get_X_y():
    '''
    Get X and y for the task of predicting whether a given
    article will get a press release.
    '''
    articles = load_articles()
    matchSample = load_matched_samples()

    article_texts = [chambers_analysis.process_article(article) for article in articles]
    matchSampe_texts = [chambers_analysis.process_article(article) for article in matchSample]

    all_texts = article_texts+matchSampe_texts[:]
    vectorizer = chambers_analysis.get_vectorizer(all_texts)

    x = vectorizer.transform(all_texts)
    #transformer = TfidfTransformer()
    #X = transformer.fit_transform(X)
    y = []
    for article in articles:
        y.append(1)
    for article in matchSampe_texts[:]:
        y.append(0)
    y = np.array(y)

    return x, y, vectorizer

def load_articles(articles_dir="1. excel files"):

    article_files = []
    for file_name in os.listdir(articles_dir):
        if file_name.endswith(".article_info.txt"):
            article_files.append(file_name)

    print "read %s articles with press lease." % len(article_files)

    articles = []
    for article_file_name in article_files:
        article = read_in_article(os.path.join(articles_dir, article_file_name))
        articles.append(article)

    return articles


def load_matched_samples(matched_dir="7. Matched samples"):
    article_files = []
    for file_name in os.listdir(matched_dir):
        article_files.append(file_name)
    print "read %s matched samples without press release." % len(article_files)
    articles = []
    for article_file_name in article_files:
        article = read_in_matched_samples(os.path.join(matched_dir, article_file_name))
        articles.append(article)
    return articles

def read_in_article(article_path):
    with open(article_path, 'rU') as input_file:
        reader = csv.reader(input_file)
        pmid, title, mesh, authors, abstract, affiliation, journal, volume = reader.next()
    return {"pmid":pmid, "title":title, "mesh":mesh, "authors":authors,
                "abstract":abstract, "affiliation":affiliation}

def read_in_matched_samples(article_path):
    with open(article_path, 'rU') as input_file:
        reader = csv.reader(input_file)
        pmid, title, journal,authors, affiliation, abstract,mesh = reader.next()
    return {"pmid":pmid, "title":title, "mesh":mesh, "authors":authors,
                "abstract":abstract, "affiliation":affiliation}

def texify_most_informative_features(vectorizer, clf, caption, n=50):
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
    X, y, vectorizer = get_X_y()
    clf = predict_PR()
    texify_most_informative_features(vectorizer, clf, "Top fifty features and associated weights for press release prediction")

if __name__ == '__main__':
    main()




