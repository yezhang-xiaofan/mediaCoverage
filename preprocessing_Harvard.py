__author__ = 'zhangye'
import os
import nltk
import pickle
import nltk
import re
from gensim.models.word2vec import Word2Vec
import numpy as np
from sets import Set
from scipy.spatial.distance import cosine
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
#This file preprocesses Oxford health research related press release into sentences
#And it label sentences as positive if it contains terms in the dictionary
root = "PR_Harvard/"
model = Word2Vec.load('/scratch/cluster/yezhang/bio_word',mmap='r')
dict = pickle.load(open("relation_terms","rb"))
stopwords = (open("english",'rb').read().splitlines())
threshold = 0.9
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
    if(len(line)==0):
          continue
    IV_DV[line] = average_vector(line)

def con_simi(u,v): 
    return 1 - cosine(u,v)
tokenizer = RegexpTokenizer(r'\w+')
def check_term(sentence):
    tokens = tokenizer.tokenize(sentence)
    for IVDV in IV_DV.keys():
        for t in tokens:
            if t in stopwords:    
               continue
            if(IV_DV[IVDV] is not None and t in model):
                if(con_simi(IV_DV[IVDV],model[t])>=threshold):
                      print "IV/DV term: "+t + "::::" + IVDV
                      return True
    return False

def process(file):
    write_file = open("Harvard_Sentence/"+file,"w")
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
   #readline for date
    #date = file.readline()
    #write_file.write(title.strip()+"\n")
    sentences  = file.readlines()

    #check main body
    for s in sentences:
         if "for immediate release" in s.lower():
             continue
         if "for more information" in s.lower():
             break
         label = 0
         for term in dict:
               pattern = r'\b'+term+r'\b'
               if(re.search(pattern,s)):
                 label = 1
                 break
         if(label==1):
            if(check_term(s)==False):
                label = 0
         write_file.write(s.strip().lower()+"\t"+str(label)+"\n")
    file.close()
    write_file.close()
for file in os.listdir(root):
    if(file.endswith(".txt")==False):
        continue
    process(file)