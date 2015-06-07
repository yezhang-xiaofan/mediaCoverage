__author__ = 'zhangye'
import os
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import  StratifiedKFold
from predictNConPR import process_article
def read_in_article(article_path):
    with open(article_path, 'rU') as input_file:
        reader = csv.reader(input_file)
        pmid, title, mesh, authors, abstract, affiliation, journal, volume = reader.next()
    return {"pmid":pmid, "title":title, "mesh":mesh, "authors":authors,
                "abstract":abstract, "affiliation":affiliation}


def load_articles(articles_dir="1. excel files",
                    news_articles_dir="6. News articles"):

    article_files = []
    for file_name in os.listdir(articles_dir):
        if file_name.endswith(".article_info.txt"):
            article_files.append(file_name)

    print "read %s articles." % len(article_files)

    articles = []
    len_abstract = 0
    len_title = 0
    for article_file_name in article_files:
        article = read_in_article(os.path.join(articles_dir, article_file_name))
        article["date"] = article_file_name.split(".")[0][:-2]
        abstract = article["abstract"]
        title = article["title"]
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

#build dictionary for tweet count label
#key is the date
#value is binary 0/1
def read_label(file = "labels0.txt"):
   labels = {}
   with open(file, 'rU') as input_file:
       for line in input_file:
            labels[line.split(",")[0]] = line.split(",")[1]
   return labels

def get_vectorizer(article_texts, max_features=5000):
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english",
                                    min_df=1,
                                    token_pattern=r"(?u)95% confidence interval|95% CI|95% ci|[a-zA-Z0-9_*\-][a-zA-Z0-9_/*\-]+",
                                    binary=False, max_features=max_features)
    vectorizer.fit(article_texts)
    return vectorizer

def get_X_y():
    labels = read_label()
    articles = load_articles()
    train_articles_labels = [(process_article(a),labels[a["date"]]) for a in articles if a["date"] in labels]
    temp = zip(*train_articles_labels)
    train_articles = list(temp[0])
    train_labels = list(temp[1])
    vectorizer = get_vectorizer(train_articles)
    X = vectorizer.transform(train_articles)
    y = map(lambda x: int(x.strip()),train_labels)
    return X,y,vectorizer
def main():
    X,y,vectorizer = get_X_y()
    lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
    kf = StratifiedKFold(y,n_folds=5,shuffle=True)
    parameters = {"C":[1.0,.1, .01, .001,0.0001]}
    clf0 = GridSearchCV(lr, parameters,scoring='roc_auc',cv=kf)
    print "fitting model..."
    clf0.fit(X,y)
    print "best auc score is: " ,clf0.best_score_
    print "done."

if __name__ == "__main__":
    main()
