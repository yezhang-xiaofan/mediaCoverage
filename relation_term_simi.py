__author__ = 'zhangye'
#This program calculates similarites between each pair of relationship terms
#plot histogram
#find a good threshold
from gensim.models.word2vec import Word2Vec
import pickle
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
print ("load model")
model = Word2Vec.load('/scratch/cluster/yezhang/bio_word',mmap='r')
stopwords = nltk.corpus.stopwords.words('english')
def average_vector(term):
    sum = np.zeros(200)
    length = 0.0
    word_list = word_tokenize(term)
    for w in word_list:
        if(w not in stopwords):
            sum += model[w.islower()]
            length += 1
    return sum/length
def con_simi(u,v):
    return 1 - cosine(u,v)


dict = pickle.load(open("relation_terms","rb"))
dict_list = list(dict)
similarities = []
for i in range(len(dict_list)-1):
    for j in range(i+1,len(dict_list)):
        similarities.append(con_simi(average_vector(dict_list[i]),average_vector(dict_list[j])))
print ("plot histogram")
plt.hist(similarities,bins=20)
plt.savefig("ca")