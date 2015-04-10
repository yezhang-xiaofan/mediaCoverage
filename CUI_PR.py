__author__ = 'zhangye'
import os
import predictPR
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.cross_validation import  StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

def association():
    X,y,vectorizer = get_X_Y()
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
    kf = StratifiedKFold(y,n_folds=5,shuffle=True)
    parameters = {"C":[1000,100,10,1.0,.1, .01, .001,0.0001]}
    clf0 = GridSearchCV(lr, parameters,scoring='roc_auc',cv=kf)
    print "fitting model..."
    clf0.fit(X,y)
    print "best auc score is: " ,clf0.best_score_
    print "done."
    texify_most_informative_features(vectorizer, clf0, "", n=50)
    '''
    num_positive = np.count_nonzero(y==1)
    num_negative = y.shape[0] - num_positive

    pos_stat = np.sum(X[:num_positive].toarray(),axis=0,dtype=np.int32)
    neg_stat = np.sum(X[num_positive:].toarray(),axis=0)
    top_pos = sorted(zip(pos_stat, vectorizer.get_feature_names()))
    top_neg = sorted(zip(neg_stat, vectorizer.get_feature_names()))
    print(top_pos[-5:])
    print(top_neg[-5:])
    '''

def get_X_Y():
    press_release_htf = load_htf()
    match_samples_htf = load_matched_sample()
    total_Doc = press_release_htf + match_samples_htf
    vectorizer = CountVectorizer(min_df=5,token_pattern=r'\b[\w^]+\b')
    X = vectorizer.fit_transform(total_Doc)
    y = np.hstack((np.ones(len(press_release_htf)),np.ones(len(match_samples_htf))*(-1)))
    return X, y, vectorizer


def load_htf(articles_dir="1. excel files"):
    article_files = []
    for file_name in os.listdir(articles_dir):
        if file_name.endswith(".htf"):
            article_files.append(file_name)

    print "read %s htf files with press lease." % len(article_files)

    articles = []
    for article_file_name in article_files:
        article = read_in_article(os.path.join(articles_dir, article_file_name))
        articles.append(article)

    return articles

def load_matched_sample(articles_dir="7. Matched samples"):
    article_files = []
    for file_name in os.listdir(articles_dir):
        if file_name.endswith(".htf"):
            article_files.append(file_name)

    print "read %s htf files without press lease." % len(article_files)

    articles = []
    for article_file_name in article_files:
        article = read_in_article(os.path.join(articles_dir, article_file_name))
        articles.append(article)
    return articles

def read_in_article(article_path):
    text = []
    file = open(article_path,'r')
    for line in file:
        text.append(line.split('|')[2])
    text = ' '.join(tuple(text))
    return text
    file.close()

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
        out_str.append("%.3f & %s & %.3f & %s \\\\" % (c1, f1.replace('^','\\textasciicircum '), c2, f2.replace('^','\\textasciicircum ')))

    #
    out_str.append(r"\end{tabular}")
    out_str.append("%s" % caption)
    out_str.append(r"\end{table}")

    feature_str = "\n".join(out_str)

    print "\n"
    print feature_str

def main():
    association()
main()
