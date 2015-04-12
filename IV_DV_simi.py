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
IVDV = open("IV_DV.txt","rb")
term_list = []
for line in IVDV:
    term_list.append(line.strip().split())
similar = []
for i in range(len(term_list)):
    for j in range(i+1,len(term_list)):
        similar.append(model.n_similarity(term_list[i],term_list[j]))
