__author__ = 'zhangye'
# predict tweet count on Reuters dataset
import pattern
import requests
from pattern import web
from pattern.web import URL
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
import os
import numpy as np
def url_to_text(input):   #scrape url, get content of news articles
     text = ''
     dom = web.Element(input)
     para = dom.by_tag('div.column1 grid8 grid-inside')[0]
     for p in para.by_tag('p'):
         for c in p.children:
             if c.type=='text': text += c.source
     return text
tweet_file = open('reuters/reuters_tweets.txt','rb')
def download_articles():
    for t in tweet_file:
        content = t.split('\t')
        PMID = content[0]
        urls = content[1]
        for u in urls.split(','):
            if('article/id' not in u and 'article/20' in u):
                f = open('reuters/news_articles/{}.txt'.format(PMID),'wb')
                f.write(url_to_text(requests.get(u).text).encode('ascii',errors='ignore'))
                f.close()

def classify():
    dir = 'reuters/news_articles/'
    file1 = 'reuters/reuters_tweets.txt'
    news = []
    labels = []
    #get labels
    label_dict = None
    with open(file1,'rb') as haha:
        label_dict = {h.split('\t')[0]:int(h.split('\t')[-1]) for h in haha}

    for file in os.listdir(dir):
        if file.endswith('.txt'):
            PMID = file.split('.')[0]
            news.append(open(dir+file,'rb').read())
            labels.append(label_dict[PMID])

    print "number of articles is {}".format(len(news))
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english")
    data = vectorizer.fit_transform(news)
    kf = KFold(len(labels),n_folds=5,shuffle=True)
    lr = Ridge()
    parameters = {"alpha":[100000,10000,1000,100,10,1.0,.1, .01, .001,0.0001,0.00001]}
    clf0 = GridSearchCV(lr,parameters,cv=kf,scoring='r2')
    clf0.fit(data,np.array(labels))
    print "best r2 score is: " + str(clf0.best_score_)

def main():
    download_articles()       #download html files
#classify()
if __name__ == "__main__":
    main()