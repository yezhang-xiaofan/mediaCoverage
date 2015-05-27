'''
Code for analyzing the Chambers et al data
'''
import random
import csv
import pdb
import os
import operator

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

import numpy as np
from numpy.linalg import norm

import pylab
#import seaborn as sns


def predict_news_article():
    '''
    It would seem we are essentially unable to predict which
    articles will get news stories, conditioned on their
    already receiving a press release! This is actually kind of
    interesting, since we *are* able to predict (certainly better
    than chance) which articles will get press release OR
    news articles. This supports the conclusions of Chambers et al:
    it seems the press release selection and process is the crucial
    thing.
    '''
    X, y, vectorizer = get_X_y()
    lr = LogisticRegression(penalty="l2", fit_intercept=True)

    parameters = {"C":[.1, .01, .001]}
    clf0 = GridSearchCV(lr, parameters, scoring='accuracy')
    print "fitting model..."
    clf0.fit(X,y)
    print "done."

    print texify_most_informative_features(vectorizer, clf0, "predictive features")

    kf = cross_validation.KFold(X.shape[0], shuffle="true", n_folds=5)
    fs, aucs = [],[]
    fold = 0
    for train, test in kf:
        clf = GridSearchCV(lr, parameters, scoring='accuracy')
        clf.fit(X[train], y[train])

        probs = clf.predict_proba(X[test])

        #aucs.append(sklearn.metrics.roc_auc_score(y[test], probs))
        cur_auc = sklearn.metrics.roc_auc_score(y[test], probs[:,1])
        aucs.append(cur_auc)
        preds = clf.predict(X[test])
        fs.append(sklearn.metrics.f1_score(y[test], preds))

        if fold == 0:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y[test], probs[:,1])
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
    pdb.set_trace()
    return clf0


def get_X_y():
    '''
    Get X and y for the task of predicting whether a given
    article will get a news article. (Recall that all articles
    in this corpus will receive press releases!)
    '''
    articles = load_articles()
    article_texts = [process_article(article) for article in articles]

    vectorizer = get_vectorizer(article_texts)
    X = vectorizer.transform(article_texts)
    #transformer = TfidfTransformer()
    #X = transformer.fit_transform(X)

    y = []
    for article in articles:
        y.append(article["has_news_article"])
    y = np.array(y)

    return X, y, vectorizer


def get_vectorizer(article_texts, max_features=50000):
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english",
                                    min_df=1,
                                    token_pattern=r"(?u)95% confidence interval|95% CI|95% ci|[a-zA-Z0-9_*\-][a-zA-Z0-9_/*\-]+",
                                    binary=False, max_features=max_features)
    vectorizer.fit(article_texts)
    return vectorizer


def load_articles(articles_dir="1. excel files",
                    news_articles_dir="6. News articles"):

    article_files = []
    for file_name in os.listdir(articles_dir):
        if file_name.endswith(".article_info.txt"):
            article_files.append(file_name)

    print "read %s articles." % len(article_files)

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
                "abstract":abstract, "affiliation":affiliation,"journal":journal}





##########################################################
# Below we define several utility methods.
#
# A fair amount of this is copy-pasta'd from the factiva
# module @TODO re-factor to avoid this duplication.
##########################################################
import re
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
                    prepend_to_words(re.sub(' +',' ',article["mesh"]).replace(",","").replace("&","").replace(" ", "-"),
                                prepend_str="MH-", split_on="\n")]


    #all_features = [article["title"], article["abstract"],
    #                prepend_to_words(article["mesh"], prepend_str="MH-")]

    return " ".join(all_features)

def _get_ranked_features(clf, vectorizer, n=50):
    c_f = sorted(zip(clf.best_estimator_.raw_coef_[0], vectorizer.get_feature_names()))

    weights_d = dict(zip(vectorizer.get_feature_names(), clf.best_estimator_.raw_coef_[0]))
    return c_f, weights_d


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