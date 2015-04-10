__author__ = 'zhangye'
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=1)
corpus = [
    'This is the first document. And it is very cool. Thank you.'
]
X = vectorizer.fit(corpus)
analyze = vectorizer.build_analyzer()
result = analyze("This is the first document. And it is very cool. Thank you.")
print result