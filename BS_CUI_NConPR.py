__author__ = 'zhangye'
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import grid_search
import cPickle as pickle
import matplotlib.pyplot as plt
def association():
    X,y,vectorizer = get_X_Y()
    n_samples = 1000
    bs_indexes = bootstrap_indexes(X,n_samples)
    w_lists = np.zeros((n_samples,X.shape[1]))
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
    kf = cross_validation.StratifiedKFold(y,n_folds=5,shuffle=True)
    parameters = {"C":[100,10,1.0,.1, .01, .001,0.0001]}
    clf0 = grid_search.GridSearchCV(lr, parameters,scoring='roc_auc',cv=kf)
    clf0.fit(X,y)
    best_C = clf0.best_params_['C']
    for i in range(n_samples):
        train_X = X[bs_indexes[i]]
        train_Y = y[bs_indexes[i]]
        lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto',C=best_C)
        lr.fit(train_X,train_Y)
        w = lr.coef_
        w_lists[i] = w
    CI_hash_pos = {}
    CI_hash_neg = {}
    mean = np.mean(w_lists,axis=0)
    std = np.std(w_lists,axis=0)
    p_lower = mean - (1.96)*std
    p_upper = mean + (1.96)*std
    sort_p_lower = sorted(zip(p_lower.tolist(),vectorizer.get_feature_names(),range(len(mean))),reverse=True)
    sort_p_upper = sorted(zip(p_upper.tolist(),vectorizer.get_feature_names(),range(len(mean))))
    texify_most_informative_features(sort_p_lower,sort_p_upper)
    #draw top features for positive instances
    for i in range(5):
        values = w_lists[:,sort_p_lower[i][2]]
        plt.hist(values,bins=20)
        plt.title(vectorizer.get_feature_names()[sort_p_lower[i][2]])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("BS_CUI_NConPR/"+vectorizer.get_feature_names()[sort_p_lower[i][2]]+"_L2.png")
        plt.clf()

    #draw top features for negative instances
    for i in range(5):
        values = w_lists[:,sort_p_upper[i][2]]
        plt.hist(values,bins=20)
        plt.title(vectorizer.get_feature_names()[sort_p_upper[i][2]])
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("BS_CUI_NConPR/"+vectorizer.get_feature_names()[sort_p_upper[i][2]]+"_L2.png")
        plt.clf()

    '''
    for x in range(50):
        index = sort_p_lower[x][2]
        CI_hash_pos[vectorizer.get_feature_names()[index]]=w_lists[:,index]
    for x in range(50):
        index = sort_p_upper[x][2]
        CI_hash_neg[vectorizer.get_feature_names()[index]]=w_lists[:,index]



    with open('BS_CUI_NConPR/CI_Hash_pos.p','wb') as f:
        pickle.dump(CI_hash_pos,f)

    with open('BS_CUI_NConPR/CI_Hash_neg.p','wb') as f:
        pickle.dump(CI_hash_neg,f)
    '''

def bootstrap_indexes(data, n_samples=1000):
    """
Given data points data, where axis 0 is considered to delineate points, return
an array where each row is a set of bootstrap indexes. This can be used as a list
of bootstrap indexes as well.
    """
    return np.random.randint(data.shape[0],size=(n_samples,data.shape[0]) )

def get_X_Y():
    articles_labels = load_articles()
    articles = [i['text'] for i in articles_labels]
    vectorizer = CountVectorizer(min_df=5,token_pattern=r'\b[\w^]+\b')
    X = vectorizer.fit_transform(articles)
    y = []
    for article in articles_labels:
        y.append(article["has_news_article"])
    y = np.array(y)
    print "read %d articles with news coverage and press release" % (np.count_nonzero(y))
    print "read %d articles without news coverage but with press release" % (len(articles)-np.count_nonzero(y))
    return X, y, vectorizer

'''
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

'''
def load_articles(articles_dir="1. excel files",
                    news_articles_dir="6. News articles"):
    article_files = []
    for file_name in os.listdir(articles_dir):
        if file_name.endswith(".htf"):
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
    text = []
    file = open(article_path,'r')
    for line in file:
        text.append(line.split('|')[2])
    text = ' '.join(tuple(text))
    return {'text':text}

def texify_most_informative_features(sort_p_lower,sort_p_upper,n=50):
    out_str = [
        r'''\begin{table}
            \caption{top 50 CUI features for news coverage prediction conditioned on press release}
            \begin{tabular}{l c|l c}

        '''
    ]
    out_str.append(r"\multicolumn{2}{c}{\emph{negative}} & \multicolumn{2}{c}{\emph{positive}} \\")
    file1 = open("BS_CUI_NConPR/positive_CUI_NC","w")
    file2 = open("BS_CUI_NConPR/negative_CUI_NC","w")
    for i in range(n):
        out_str.append("%.3f & %s & %.3f & %s \\\\" % (sort_p_upper[i][0], sort_p_upper[i][1].replace('^','\\textasciicircum '),sort_p_lower[i][0], sort_p_lower[i][1].replace('^','\\textasciicircum ')))
        file1.write(str(sort_p_lower[i][0])+" "+sort_p_lower[i][1]+"\n")
        file2.write(str(sort_p_upper[i][0])+" "+sort_p_upper[i][1]+"\n")
    file1.close()
    file2.close()

    out_str.append(r"\end{tabular}")
    out_str.append(r"\end{table}")

    feature_str = "\n".join(out_str)

    print "\n"
    print feature_str

def main():
    association()

main()