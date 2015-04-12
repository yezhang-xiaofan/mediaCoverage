__author__ = 'zhangye'
import os
import nltk
import pickle
import nltk
import re
from gensim.models.word2vec import Word2Vec
import numpy as np
from sets import Set
from nltk.tokenize import word_tokenize
#This file preprocesses Oxford health research related press release into sentences
#And it label sentences as positive if it contains terms in the dictionary
root = "PressRelease_Oxford/"
model = Word2Vec.load('/scratch/cluster/yezhang/bio_word',mmap='r')
dict = pickle.load(open("relation_terms","rb"))
stopwords = nltk.corpus.stopwords.words('english')
#more_stopwords = ["have","more"]
#stopwords += more_stopwords
threshold = 0.25     #if the distance between any of the token in the sentence and any of the
#IV/DV is less than threshold, then label as 1
#remove stopwords
for s in stopwords:
    if s in dict:
        del dict[s]

#load IV/DV terms
file = open("IV_DV.txt",'rb')
IV_DV = {}
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

for line in file:
    line = line.strip()
    IV_DV[line] = average_vector(line)

def check_term(sentence):
    tokens = word_tokenize(sentence)
    for IVDV in IV_DV.keys():
        for t in tokens:
            if t in stopwords:
                continue
            if(IV_DV[IVDV] is not None and t in model):
                if(model.similarity(IV_DV[IVDV],t)<=threshold):
                    return True
    return False

def process(file):
    write_file = open("PR_Oxford_Sentence/"+file,"w")
    file = open(root+file,'r')
    first_line = file.readline()
    #check title
    title = first_line.split('|')[0]
    label = 0
    for term in dict:
        pattern = r'\b'+term+r'\b'
        if(re.search(pattern,title)):
            #print term
                #print title
            label = 1
            break
    if(label==1):
        if(check_term(title)==False):
            label = 0
    write_file.write(title+" "+str(label)+"\n")
    date = file.readline()
    #write_file.write(title.strip()+"\n")
    sentences = []

    #check main body
    line = file.readline()
    while (line):
        cur_sen = nltk.sent_tokenize(line)
        sentences += cur_sen
        line = file.readline()
    for s in sentences:
        label = 0
        for term in dict:
            pattern = r'\b'+term+r'\b'
            if(re.search(pattern,s)):
                label = 1
                break
        if(label==1):
            if(check_term(s)==False):
                label = 0
        write_file.write(s.strip()+"\t"+str(label)+"\n")
    file.close()
    write_file.close()
for file in os.listdir(root):
    if(file.endswith(".txt")==False):
        continue
    process(file)