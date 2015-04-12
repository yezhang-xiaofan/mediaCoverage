__author__ = 'zhangye'
#This program calculates similarites between each pair of IV/DV terms
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
def average_vector(term):
    sum = np.zeros(200)
    length = 0.0
    word_list = word_tokenize(term)
    for w in word_list:
	   if(w.lower() in model):
	       sum += model[w.lower()]
           length += 1
    if length==0.0:
       print term
       return None
    return sum/length
def con_simi(u,v):
    return 1 - cosine(u,v)

IVDV = open("IV_DV.txt","rb")
term_list = []
for line in IVDV:
   term_list.append(line.strip())
similar = []
for i in range(len(term_list)):
    vector1 = average_vector(term_list[i])
    if(vector1 is None):
        print term_list[i]
        continue
    for j in range(i+1,len(term_list)):
       vector2 = average_vector(term_list[j])
       if(vector2 is None):
            continue
       similar.append(con_simi(vector1,vector2))
pickle.dump(similar,open("IVDV_simi","wb"))
