__author__ = 'zhangye'
#this file create LDA data and label files for JAMA dataset
from factiva_model import process_file
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
def get_X_y():
    X,y,interest = process_file("jama/jama_article_info.csv","jama/jama_pmids.txt_matched_articles_filtered.csv")
    vectorizer = CountVectorizer(stop_words="english",
                                    min_df=2,
                                    token_pattern=r"(?u)95% confidence interval|95% CI|95% ci|[a-zA-Z0-9_*\-][a-zA-Z0-9_/*\-]+",
                                    binary=False, max_features=50000)
    X = vectorizer.fit_transform(X)
    return X,np.array(y),vectorizer
if __name__ == "__main__":
    root = "jama/"
    out_data = open(root+"LDA_data",'wb')
    out_label = open(root+"LDA_label",'wb')
    X, y, vectorizer = get_X_y()
    y[y==-1] = 0
    X = X.tocsr()
    indices = X.indices
    indptr = X.indptr
    data = X.data
    for i in range(X.shape[0]):
        col_index = indices[indptr[i]:indptr[i+1]]
        elements = data[indptr[i]:indptr[i+1]]
        out_data.write(str(elements.sum())+" ")
        for j in range(len(elements)):
            out_data.write(str(col_index[j])+":"+str(elements[j])+" ")
        out_data.write("\n")
        out_label.write(str(y[i])+"\n")
    out_data.close()
    out_label.close()