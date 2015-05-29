__author__ = 'zhangye'
#this file classify chambers sentences with the help of Harvard and Oxford labeled sentences
import os
import xlrd
Chambers_sentence = "Chambers_sen/"
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
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
#from relation_term_simi import con_simi
#read Chambers sentences files
#build a hash table for IV/DV keys is the index of documents
directory = 'Chambers_sen'
dir2 = "IV_DV"
Documents = []
IVDV = {}
i = 0
k = 0
numToName = {}
for file in os.listdir(directory):
    if(file.endswith(".txt")):
    	cur = open(directory+"/"+file,'rb')
    	Documents.append(cur.readlines())
        numToName[k] = file        #map index to file name
        k+= 1
stopwords = (open("english",'rb').read().splitlines())
#get average vector of a single IV/DV term (could be a single word or phrase)
#return the sum and number of word in this IV/DV term
def average_vector(term):
    sum = np.zeros(200)
    length = 0.0
    term = term.split("\t")
    for w in term:
	   if(w.lower() in model and w.lower() not in stopwords):
	       sum += model[w.lower()]
           length += 1
    return (sum,length)

#read in the IV/DV terms
#store IV/DV in the IVDV hash tabel
#key is the filename
#values include number of tokens and sum vector of all tokens in IV/DV
for file in os.listdir(dir2):
    if(file.endswith(".txt")):
        IV_DV_file = open(dir2+"/"+file,'rb')
        lines = IV_DV_file.readline()
        IVDV[file] = average_vector(lines)


#extract similarity features
def check_simi(s1,s2):
    tokenizer = RegexpTokenizer(r'\w+')
    t1 = filter(lambda x:x not in stopwords and x in model,tokenizer.tokenize(s1.lower()))
    t2 = filter(lambda x:x not in stopwords and x in model,tokenizer.tokenize(s2.lower()))
    temp = model.n_similarity(t1,t2)
    if(math.isnan(temp)):
        return float('inf')
    else:
        return temp


#calcualte the closest similarity between the token in the sentence and average IV/DV in the training data
def sen_IVDV_simi(sentence,average):
    max_simi = 0.0
    for s in sentence:
        if s in model and s not in stopwords:
            temp = 1 - cosine(model[s],average)
            if(temp>max_simi):
                max_simi = temp
    return max_simi


simi = []
pos = []

#j = 0
for d in Documents:
    temp_simi = []
    temp_pos = []
    for i in range(len(d)):
        if(i==0 or i==1 or i ==2):
            temp_pos.append(1)
            temp_simi.append(1)
            continue
        temp_simi.append(max([check_simi(d[i],d[0]),check_simi(d[i],d[1]),check_simi(d[i],d[2])]))
        temp_pos.append(0)
    simi.append(temp_simi)
    pos.append(temp_pos)

sentences = []
y = []
num_Doc = len(Documents)
#bag of words
#train
vectorizer = CountVectorizer(ngram_range=[1,2],min_df = 1,stop_words="english")
dataset = {}
i = 0

'''
for d in Documents:
    for i in range(len(d)):
        sen = d[i]

    for sen in d:
        y.append(int(sen.strip()[-1]))
        sentences.append((sen.strip()[:-1]))
'''
labeled_sen = []
labeled_y = []

#read the Oxford press release labeled by the dictionary terms

Ox_simi = []
Ox_pos = []
for file_name in os.listdir("PR_Oxford_Sentence"):
    if(not file_name.endswith(".txt")):
        continue
    temp = open("PR_Oxford_Sentence/"+file_name,'rb')
    temp = temp.readlines()
    for i in range(len(temp)):
        line = temp[i]
        line = line.strip()
        label = int(line[-1].strip())
        sen = line[:-1]
        labeled_sen.append(sen.strip())
        if(i<=3):
            labeled_y.append(1)
            Ox_simi.append(1)
            Ox_pos.append(1)
        else:
            labeled_y.append(0)
            Ox_simi.append(max([check_simi(temp[i],temp[0]),check_simi(temp[i],temp[1]),check_simi(temp[i],temp[2])]))
            Ox_pos.append(0)
   # temp.close()
labeled_sen_Hv= []
labeled_y_Hv= []
Hv_simi = []
Hv_pos = []
for file_name in os.listdir("Harvard_Sentence"):
    if(not file_name.endswith(".txt")):
        continue
    temp = open("Harvard_Sentence/"+file_name,'rb')
    temp = temp.readlines()
    for i in range(len(temp)):
        line = temp[i]
        line = line.strip()
        #label = int(line[-1].strip())
        sen = line[:-1]
        labeled_sen_Hv.append(sen.strip())
        if(i<=3):
            Hv_simi.append(1)
  	    labeled_y_Hv.append(1)
            Hv_pos.append(1)
        else:
            labeled_y_Hv.append(0)
            Hv_simi.append(max([check_simi(temp[i],temp[0]),check_simi(temp[i],temp[1]),check_simi(temp[i],temp[2])]))
            Hv_pos.append(0)
#    temp.close()
parameters = [.1,0.01,.001,.0001]
#parameters = [.001]
kf = cross_validation.KFold((num_Doc),n_folds=5,shuffle=False)
#lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')
label_weight = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#label_weight = [0.1]


for l in label_weight:
    best_p = 0.0
    best_r = 0.0
    for p in parameters:
        #lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto',C=p)
        mean = []
        for train_index, test_index in kf:
            #extract IV/DV from training data and get the average vector for IV/DV in the training set
            length = 0.0
            sum_vec = np.zeros(200)
            for t in train_index:
                nameofFile = numToName[t]
                sum_vec += IVDV[nameofFile][0]
                length += IVDV[nameofFile][1]
            average = sum_vec/length

            sentences = [sen.strip()[:-1] for t in train_index for sen in Documents[t]]
            labels = [int(sen.strip()[-1]) for t in train_index for sen in Documents[t]]
            train_sentence = sentences + labeled_sen + labeled_sen_Hv

            train_label = labels + labeled_y +labeled_y_Hv
            test_sentence = [sen.strip()[:-1] for t in test_index for sen in Documents[t]]
            test_label = [int(sen.strip()[-1])for t in test_index for sen in Documents[t]]
            train_sentence_sparse = vectorizer.fit_transform(train_sentence)
            lr = SGDClassifier(loss="log",fit_intercept=True,class_weight='auto',alpha=p,shuffle=False,n_iter=np.ceil((10**6)/(len(train_label))))
            test_sentence_sparse = vectorizer.transform(test_sentence)
	    train_data = train_sentence_sparse
            test_data = test_sentence_sparse
           
            #insert features based on similarity between token in the target sentence and IV/DV
            '''
            doc_terms_train = vectorizer.inverse_transform(train_sentence_sparse)
            doc_terms_test = vectorizer.inverse_transform(test_sentence_sparse)
            sen_IVDV_train = [sen_IVDV_simi(d,average) for d in doc_terms_train]
            sen_IVDV_test = [sen_IVDV_simi(d,average) for d in doc_terms_test]
            '''
            ins_weight = np.ones(len(sentences))
            ins_weight = np.concatenate((ins_weight,np.ones(len(labeled_y+labeled_y_Hv))*l))
            	    
            #insert similarity features
            #for each sentence, compare it with title and first two sentences
                       
            simi_fea_train = [s for t in train_index for s in simi[t]] + Ox_simi + Hv_simi
            simi_fea_test = [s for t in test_index for s in simi[t]]
            train_data = hstack([train_sentence_sparse,np.array(simi_fea_train).reshape((len(train_label),1))])
            test_data = hstack([test_sentence_sparse,np.array(simi_fea_test).reshape((len(test_label),1))])
            #insert position features
	    pos_fea_train = [s for t in train_index for s in pos[t]] + Ox_pos + Hv_pos
            pos_fea_test = [s for t in test_index for s in pos[t]]
            train_data = hstack([train_data,np.array(pos_fea_train).reshape((len(train_label),1))])
            test_data = hstack([test_data,np.array(pos_fea_test).reshape((len(test_label),1))])
            
            #insert sentence IV/DV similarity feature
            '''
            train_data = hstack([train_data,np.array(sen_IVDV_train).reshape((len(train_label),1))])
            test_data = hstack([test_data,np.array(sen_IVDV_test).reshape((len(test_label),1))])
            '''
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
