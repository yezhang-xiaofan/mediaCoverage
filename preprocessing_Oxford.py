__author__ = 'zhangye'
import os
import nltk
import pickle
import nltk
import re
#This file preprocesses Oxford health research related press release into sentences
#And it label sentences as positive if it contains terms in the dictionary
root = "PressRelease_Oxford/"
dict = pickle.load(open("relation_terms","rb"))
stopwords = (open("english",'rb').read().splitlines())
#stopwords = set(haha.readlines())
#stopwords = nltk.corpus.stopwords.words('english')
#more_stopwords = ["have","more"]
#stopwords += more_stopwords
for s in stopwords:
    if s in dict:
        del dict[s]
def process(file):
    write_file = open("PR_Oxford_Sentence/"+file,"w")
    file = open(root+file,'r')
    first_line = file.readline()
    title = first_line.split('|')[0]
    label = 0
    for term in dict:
        pattern = r'\b'+term+r'\b'
        if(re.search(pattern,title)):
            #print term
            #print title
            label = 1
            break
    write_file.write(title+" "+str(label)+"\n")
   #readline for date
    date = file.readline()
    #write_file.write(title.strip()+"\n")
    sentences  = file.readlines()
   # while (line):
    #    cur_sen = line
     #   sentences += cur_sen
      #  line = file.readline()
    for s in sentences:
        label = 0
        for term in dict:
            pattern = r'\b'+term+r'\b'
            if(re.search(pattern,s)):
                #print term
                #print s
                label = 1
                break
        write_file.write(s.strip()+" "+str(label)+"\n")
    file.close()
    write_file.close()
for file in os.listdir(root):
    if(file.endswith(".txt")==False):
        continue
    process(file)
