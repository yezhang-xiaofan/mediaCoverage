__author__ = 'zhangye'
import os
import nltk
import pickle
#This file preprocesses Oxford health research related press release into sentences
#And it label sentences as positive if it contains terms in the dictionary
root = "PressRelease_Oxford/"
dict = pickle.load(open("relation_terms","rb"))
def process(file):
    write_file = open("PR_Oxford_Sentence/"+file+".txt","w")
    file = open(root+file,'r')
    first_line = file.readline()
    title = first_line.split('|')[0]
    date = file.readline()
    write_file.write(title.strip()+"\n")
    sentences = []
    line = file.readline()
    while (line):
        cur_sen = nltk.sent_tokenize(line)
        sentences += cur_sen
        line = file.readline()
    for s in sentences:
        label = 0
        for term in dict:
            if(term in s):
                label = 1
                break
        write_file.write(s+" "+str(label)+"\n")
    file.close()
    write_file.close()
for file in os.listdir(root):
    if(file.endswith(".txt")==False):
        continue
    process(file)