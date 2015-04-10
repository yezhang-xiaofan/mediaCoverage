__author__ = 'zhangye'
import os
import nltk
import pickle
import nltk
import re
#This file preprocesses Oxford health research related press release into sentences
#And it label sentences as positive if it contains terms in the dictionary
root = "PR_Harvard/"
dict = pickle.load(open("relation_terms","rb"))
stopwords = nltk.corpus.stopwords.words('english')
more_stopwords = ["have","more"]
stopwords += more_stopwords
for s in stopwords:
    if s in dict:
        del dict[s]
def process(file):
    write_file = open("Harvard_Sentence/"+file,"w")
    file = open(root+file,'r')
    title = file.readline()
    #title = first_line.split('|')[0]
    label = 0
    for term in dict:
        pattern = r'\b'+term+r'\b'
        if(re.search(pattern,title)):
            #print term
            #print title
            label = 1
            break
    write_file.write(title+" "+str(label)+"\n")
    #date = file.readline()
    #write_file.write(title.strip()+"\n")
    sentences = []
    line = file.readline()
    while (line):
        if(line.startswith("For immediate release:")):
            line = file.readline()
            continue
        if(line.startswith("For more information:")):
            break
        cur_sen = nltk.sent_tokenize(line)
        sentences += cur_sen
        line = file.readline()
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