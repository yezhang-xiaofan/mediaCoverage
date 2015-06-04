__author__ = 'zhangye'
#this program generates input data for SLDA for Chambers dataset
import predictPR
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import chambers_analysis
import predictNConPR
root = "BS_NConPR/"
out_data = open(root+"LDA_data",'wb')
out_label = open(root+"LDA_label",'wb')

def get_vectorizer(article_texts, max_features=5000):
    vectorizer = CountVectorizer(stop_words="english",
                                    min_df=2,
                                    token_pattern=r"(?u)95% confidence interval|95% CI|95% ci|[a-zA-Z0-9_*\-][a-zA-Z0-9_/*\-]+",
                                    binary=False, max_features=max_features)
    vectorizer.fit(article_texts)
    return vectorizer

def get_X_y_PR():
    '''
    Get X and y for the task of predicting whether a given
    article will get a press release.
    '''
    articles,journal_pos = predictPR.load_articles()
    matchSample,journal_neg = predictPR.load_matched_samples()
    article_texts = [chambers_analysis.process_article(article) for article in articles]
    matchSampe_texts = [chambers_analysis.process_article(article) for article in matchSample]

    all_texts = article_texts+matchSampe_texts[:]
    vectorizer = get_vectorizer(all_texts)

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

def get_X_y_NC():
    '''
    Get X and y for the task of predicting whether a given
    article will get a news article. (Recall that all articles
    in this corpus will receive press releases!)
    '''
    articles = predictNConPR.load_articles()
    article_texts = [predictNConPR.process_article(article) for article in articles]

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

def main():
    #get data for press release
    X, y, vectorizer = get_X_y_PR()

    #get data for news coverage
    X, y, vectorizer = get_X_y_NC()
    X = X.toarray()
    for i in range(X.shape[0]):
        index = np.nonzero(X[i])
        out_data.write(str(index[0].shape[0])+" ")
        for j in index[0]:
            out_data.write(str(j)+":"+str(X[i][j])+" ")
        out_data.write("\n")
        out_label.write(str(y[i])+"\n")
    out_data.close()
    out_label.close()

if __name__ == "__main__":
    main()


