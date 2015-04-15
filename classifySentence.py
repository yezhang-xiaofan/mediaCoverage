__author__ = 'zhangye'
#this file classify chambers sentences with the help of Harvard and Oxford labeled sentences
import os
import xlrd
Chambers_sentence = "Chambers_sen/"
import os
from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import grid_search
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from gensim.models.word2vec import Word2Vec
model = Word2Vec.load('/scratch/cluster/yezhang/bio_word',mmap='r')
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import coo_matrix, hstack
import math
#from relation_term_simi import con_simi
#read Chambers sentences files
directory = 'Chambers_sen'
Documents = []
for file in os.listdir(directory):
    cur = open(directory+"/"+file,'rb')
    Documents.append(cur.readlines())

#extract additional features
stopwords = (open("english",'rb').read().splitlines())
def check_simi(s1,s2):
    tokenizer = RegexpTokenizer(r'\w+')
    t1 = filter(lambda x:x not in stopwords and x in model,tokenizer.tokenize(s1.lower()))
    t2 = filter(lambda x:x not in stopwords and x in model,tokenizer.tokenize(s2.lower()))
    temp = model.n_similarity(t1,t2)
    if(math.isnan(temp)):
        return float('inf')
    else:
        return temp


simi = []

#j = 0
for d in Documents:
    for i in range(len(d)):
        if(i==0 or i==1 or i ==2):
            simi.append(1)
            continue
        temp0 = check_simi(d[i],d[0])
        temp1 = check_simi(d[i],d[1])
        temp2 = check_simi(d[i],d[2])
        if(temp0>=.9 or temp1>=.9 or temp2 >=.9):
            simi.append(1)
        else:
            simi.append(0)

sentences = []
y = []
num_Doc = len(Documents)
#bag of words
#train
vectorizer = CountVectorizer(ngram_range=[1,2],min_df = 1,stop_words="english")
dataset = {}
i = 0
for d in Documents:
    for i in range(len(d)):
        sen = d[i]

    for sen in d:
        y.append(int(sen.strip()[-1]))
        sentences.append((sen.strip()[:-1]))

labeled_sen = []
labeled_y = []
'''
#read the Oxford press release labeled by the dictionary terms

for file_name in os.listdir("PR_Oxford_Sentence"):
    if(not file_name.endswith(".txt")):
        continue
    temp = open("PR_Oxford_Sentence/"+file_name)
    for line in temp:
        line = line.strip()
        label = int(line[-1].strip())
        sen = line[:-1]
        labeled_sen.append(sen.strip())
        labeled_y.append(label)
    temp.close()
'''

labeled_sen_Hv= []
labeled_y_Hv= []
'''
for file_name in os.listdir("Harvard_Sentence"):
    if(not file_name.endswith(".txt")):
        continue
    temp = open("Harvard_Sentence/"+file_name)
    for line in temp:
        line = line.strip()
        label = int(line[-1].strip())
        sen = line[:-1]
        labeled_sen_Hv.append(sen.strip())
        labeled_y_Hv.append(label)
    temp.close()
'''

parameters = [.1,0.01,.001,.0001]
#parameters = [.001]
kf = cross_validation.KFold(len(y),n_folds=5,shuffle=True)
#lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
#label_weight = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
label_weight = [0.1]
for l in label_weight:
    best_p = 0.0
    best_r = 0.0
    for p in parameters:
        #lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto',C=p)
        mean = []
        for train_index, test_index in kf:
            train_sentence = list(itemgetter(*train_index)(sentences)) + labeled_sen + labeled_sen_Hv
            train_label = list(itemgetter(*train_index)(y)) + labeled_y +labeled_y_Hv
            test_sentence = list(itemgetter(*test_index)(sentences))
            lr = linear_model.SGDClassifier(loss='log',penalty='l2',fit_intercept=True,class_weight='auto',alpha=p,n_iter=np.ceil((10**6)/len(train_sentence)))
            train_sentence_sparse = vectorizer.fit_transform(train_sentence)
            test_sentence_sparse = vectorizer.transform(test_sentence)
            test_label = list(itemgetter(*test_index)(y))
            ins_weight = np.ones(len(train_index))
            ins_weight = np.concatenate((ins_weight,np.ones(len(labeled_y+labeled_y_Hv))*l))
            train_data = hstack(train_sentence_sparse,itemgetter(*train_index)(simi))
            test_data = hstack(test_sentence_sparse,itemgetter(*test_index)(simi))
            lr.fit(train_data,np.array(train_label),sample_weight=ins_weight)
            predict = lr.predict(test_data)

            f1 = f1_score(test_label,predict)
            mean.append(f1)
        if(sum(mean)/len(mean)>best_r):
            best_p = p
            best_r = sum(mean)/len(mean)
    print("weight: "+str(l)+" "+str(best_p)+" "+str(best_r)+"\n")

'''
#test
test_labels = []
test_sentences= []
for d in test_Doc:
    for sen in d:
        test_labels.append(int(sen.strip()[-1]))
        test_sentences.append((sen.strip()[:-1]))
test_X = vectorizer.transform(test_sentences)
print "number of test documents: " , len(test_Doc)
print "number of test sentences: " , len(test_sentences)
predict_y = lr.predict(test_X)

print "precision: " + str(precision_score(test_labels,predict_y))
print "recall: " + str(recall_score(test_labels,predict_y))
print "f1 score: " + str(f1_score(test_labels,predict_y))
'''