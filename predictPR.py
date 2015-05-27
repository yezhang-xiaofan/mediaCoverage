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
from os.path import basename
from sklearn.cross_validation import  StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import auc_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif,SelectKBest
import numpy as np
from numpy.linalg import norm
import re
import pylab
from textblob import TextBlob
def predict_PR():

    #feature selection
    X, y, vectorizer = get_X_y()
    #selector = SelectKBest(f_classif,1000)
    selector = SelectPercentile(f_classif,percentile=100)
    selector.fit(X,y)
    X = selector.transform(X)
    best_indices = selector.get_support(indices=True)
    best_features = np.array(vectorizer.get_feature_names())[best_indices]

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
        #average_precision_score(y[test],probs)
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
            plt.plot(precision,recall)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0,1.05])
            plt.xlim([0.0,1.0])
            plt.title('Precision-Recall curve for press release prediction with %d vocabulary size' \
                      %len(best_indices))
            plt.show()
        fold += 1
        '''

        if fold == 0:
            fpr, tpr, thresholds = roc_curve(y[test], probs)
            pylab.clf()
            fout = "PR/roc"

            pylab.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % cur_auc)
            pylab.plot([0,1], [0,1], 'k--')
            pylab.xlim((-0.025,1.025))
            pylab.ylim((-0.025,1.025))
            pylab.xlabel("false positive rate")
            pylab.ylabel("true positive rate")
            pylab.title("ROC curve for press release prediction(area = %0.2f) " % cur_auc)
            pylab.tight_layout()
            pylab.savefig(fout)
        fold += 1

    #print "average auc: %s" % (sum(aucs)/float(len(aucs)))
    #print "average fs: %s" % (sum(fs)/float(len(fs)))
    #print "average recall: %s" % (sum(rec)/float(len(rec)))
    #print "average precision: %s" % (sum(prec)/float(len(prec)))
    #print "ABOUT TO RETURN"
    #pdb.set_trace()
    texify_most_informative_features(best_features,vectorizer,clf0,caption="",n=50)
    return clf0


def get_X_y():
    '''
    Get X and y for the task of predicting whether a given
    article will get a press release.
    '''
    articles,journal_pos = load_articles()
    matchSample,journal_neg = load_matched_samples()

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
    inspect = "results"
    pattern = re.compile(r'\bresults\b')
    target = open(inspect+"_abstract", 'w')
    journal_pos = {}
    len_abstract = 0
    len_title = 0
    num_article = 0
    for article_file_name in article_files:
        article = read_in_article(os.path.join(articles_dir, article_file_name))
        abstract = article["abstract"]
        title = article["title"]
        len_abstract += len(TextBlob(abstract).words)
        len_title += len(TextBlob(title).words)
        #pull out abstracts containing certain words
        if(pattern.search(article["abstract"])):
            target.write(article["abstract"]+"\n"+article["journal"]+"\n")
            target.write("\n")
            if(journal_pos.has_key(article["journal"])):
                journal_pos[article["journal"]] += 1
            else:
                journal_pos[article["journal"]] = 1

        articles.append(article)
    target.close()
    print "total length of titles in positive instances: "+ str(len_title)
    print "total length of abstracts in positive instances: "+ str(len_abstract)
    return articles,journal_pos


def load_matched_samples(matched_dir="7. Matched samples"):
    article_files = []
    for file_name in os.listdir(matched_dir):
        if file_name.endswith(".csv"):
            article_files.append(file_name)
    print "read %s matched samples without press release." % len(article_files)
    articles = []
    inspect = "results"
    pattern = re.compile(r'\bresults\b')
    target = open(inspect+"_abs_neg", 'w')
    journal_neg = {}
    len_abstract = 0
    len_title = 0
    for article_file_name in article_files:
        article = read_in_matched_samples(os.path.join(matched_dir, article_file_name))
        abstract = article["abstract"]
        title = article["title"]
        len_abstract += len(TextBlob(abstract).words)
        len_title += len(TextBlob(title).words)
        #pull out abstracts of negative examples containing certain words
        if(pattern.search(article["abstract"])):
            target.write(article["abstract"]+"\n")
            target.write(article["journal"]+"\n")
            target.write("\n")
            if(journal_neg.has_key(article["journal"])):
                journal_neg[article["journal"]] += 1
            else:
                journal_neg[article["journal"]] = 1
        articles.append(article)
    print "total length of titles in negative instances: "+ str(len_title)
    print "total length of abstracts in negative instances: "+ str(len_abstract)
    return articles,journal_neg

def read_in_article(article_path):
    with open(article_path, 'rU') as input_file:
        reader = csv.reader(input_file)
        pmid, title, mesh, authors, abstract, affiliation, journal, volume = reader.next()
    return {"pmid":pmid, "title":title, "mesh":mesh, "authors":authors,
                "abstract":abstract, "affiliation":affiliation,"block":basename(input_file.name)[0:9],"journal":journal}

def read_in_matched_samples(article_path):

    with open(article_path, 'rU') as input_file:
        reader = csv.reader(input_file)
        pmid, title, journal,authors, affiliation, abstract,mesh = reader.next()
    return {"pmid":pmid, "title":title, "mesh":mesh, "authors":authors,
                "abstract":abstract, "affiliation":affiliation,"block":basename(input_file.name)[0:9],"journal":journal}

def texify_most_informative_features(best_features,vectorizer, clf, caption, n=50):
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
    print "%d most informative features:" % (n, )
    out_str = [
        r'''\begin{table}
            \caption{top 50 features for press release prediction}
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
    #X, y, vectorizer = get_X_y()
    predict_PR()
    #texify_most_informative_features(vectorizer, clf, "Top fifty features and associated weights for press release prediction")

if __name__ == '__main__':
    main()




