__author__ = 'zhangye'
#this program predicts news coverage conditioned on press release
#use chambers dataset

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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
import predictPR

import numpy as np
from numpy.linalg import norm

import pylab
def predictNConPR():
    X, y, vectorizer = get_X_y()
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight="auto")

    parameters = {"C":[1.0,.1, .01, .001,0.0001]}
    clf0 = GridSearchCV(lr, parameters)
    print "fitting model..."
    clf0.fit(X,y)
    print "done."

#print texify_most_informative_features(vectorizer, clf0, "predictive features")

    kf = cross_validation.KFold(X.shape[0], shuffle=True, n_folds=5)
    fs, aucs = [],[]
    fold = 0
    complete_X = X.tocsr()
    for train, test in kf:
        clf = GridSearchCV(lr, parameters,scoring="roc_auc")
        clf.fit(complete_X[train,:].tocoo(), y[train])

        probs = clf.predict_proba(complete_X[test,:])
        #aucs.append(sklearn.metrics.roc_auc_score(y[test], probs))
        cur_auc = auc_score(y[test], probs[:,1])
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
    print "read %d articles with news coverage and press release" % (np.count_nonzero(y))
    print "read %d articles without news coverage but with press release" % (len(articles)-np.count_nonzero(y))
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
def main():
    X, y, vectorizer = get_X_y()
    clf = predictNConPR()
    chambers_analysis.texify_most_informative_features(vectorizer,clf,"haha")
main()