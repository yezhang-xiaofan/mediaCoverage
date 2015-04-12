__author__ = 'zhangye'
#This program calculates similarites between each pair of relationship terms
#plot histogram
#find a good threshold
from gensim.models.word2vec import Word2Vec
import pickle
import pdb
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
print ("load model")
model = Word2Vec.load('/scratch/cluster/yezhang/bio_word',mmap='r')
#stopwords = nltk.corpus.stopwords.words('english')
stopwords = ["is","that"]
def average_vector(term):
    sum = np.zeros(200)
    length = 0.0
    word_list = word_tokenize(term)
    for w in word_list:
        if(w not in stopwords):
         #   pdb.set_trace()
	    if(w.lower() in model):
	    	sum += model[w.lower()]
            	length += 1
	    #else:
                #print ("not in model: "+term)
    if length==0.0:
       print term	
       return None
    return sum/length
def con_simi(u,v):
    return 1 - cosine(u,v)


dic = pickle.load(open("relation_terms","rb"))
dict_list = list(dic)
similarities = []
for i in range(len(dict_list)-1):
    vector1 = average_vector(dict_list[i])
    if(vector1 is None):
        print dict_list[i]
        continue
    for j in range(i+1,len(dict_list)):
       vector2 = average_vector(dict_list[j])
       if(vector2 is None):
            continue
       similarities.append(con_simi(vector1,vector2))
#print(similarities)
print ("plot histogram")
#plt.hist(similarities,bins=20)
with open("simi",'wb') as f:
    pickle.dump(similarities,f)
#plt.savefig("ca")
