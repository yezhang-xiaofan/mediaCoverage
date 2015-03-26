__author__ = 'zhangye'
#this file predicts relationship and its level in sentence level
import xlrd
import re
import os
from nltk.tokenize import sent_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
import codecs
import numpy as np
import unicodedata
import pickle
def convert(str1):
    if(type(str1) is unicode):
        str1 = str1.encode('ascii','ignore')
        str1 = str1.strip('.')
        str1 = str1.lower()
        temp =  re.split(r'\.\. \.\.|,',str1)
        return [a for a in temp if a and a.strip().isspace()==False]

sentences = []
y = []
threshold = 0
level = []
sen_file = open("sentence.txt","w")
ignore = ["posted on","word count","sentence","title"]
dic = set()
def is_ignore(text):
    for str in ignore:
        if(str in text):
            return 1
    return 0
for file_name in os.listdir("1. excel files"):
    if file_name.endswith(".xls"):
        #print file_name
        book = xlrd.open_workbook("1. excel files/"+file_name)
        first_sheet = book.sheet_by_index(0)
        relation = first_sheet.cell(124,5).value
        code =  first_sheet.cell(125,5).value
        #if(code==0.0):
            #print file_name
        terms = first_sheet.cell(129,5).value
        if(terms!=-9):
            print terms
            terms = convert(terms)
            #sen_file.write(file_name+" "+', '.join(terms)+" \n")
        filename = file_name.split('.')[0]
        f = codecs.open("5. Press releases/"+filename[:-2]+".txt",'r',encoding='utf-8')
        text = f.readlines()
        text = [i.encode('ascii','ignore') for i in text]
        #sents = sent_tokenize(text)
        for line in text:
            if(line.strip().isspace()):
                continue
            #print line
            line = line.lower()
            if(is_ignore(line)):
                continue
            sentences.append(line)
            #code is equal to zero, no relationship
            if(code<=threshold):
                #level.append(0)
                y.append(0)
                continue
            #code is non zero,
            if relation.encode('ascii','ignore').lower() in line:
                level.append(code)
                y.append(1)
                #print "sen "+line
                sen_file.write(file_name+" "+line+"\n")
            elif(terms!=-9):
                flag = 0
                for i in terms:
                    if(i=="as above"):
                        continue
                    dic.add(i)
                    if i in line:
                        y.append(1)
                        flag = 1
                        level.append(code)
                        #print "sen "+line
                        sen_file.write(file_name+" "+line+"\n")
                        #print line
                        break
                if(flag==0):
                    y.append(0)
            else:
                y.append(0)
sen_file.close()
vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english",
                                    min_df=2,
                                    token_pattern=r"(?u)[a-zA-Z0-9-_/*][a-zA-Z0-9-_/*]+\b",
                                    binary=False, max_features=50000)
sentences = vectorizer.fit_transform(sentences)
relation_index = np.nonzero(y)
relation_sentence = sentences[relation_index]
kf = cross_validation.StratifiedKFold(y,n_folds=5,shuffle=True)
lr = LogisticRegression(penalty="l2", fit_intercept=True)
parameters = {"C":[10,1, .1, .01, .001]}
clf0 = GridSearchCV(lr, parameters, scoring='f1',cv=kf)
print "fitting model..."
clf0.fit(sentences,y)
print clf0.grid_scores_

level = np.array(level)
threshold_index = np.nonzero(level>3)
y1 = np.zeros(len(level))
y1[threshold_index] = 1
kf1 = cross_validation.StratifiedKFold(y1,n_folds=5,shuffle=True)
lr1 = LogisticRegression(penalty="l2",fit_intercept=True)
clf1 = GridSearchCV(lr1, parameters, scoring='f1',cv=kf1)
clf1.fit(relation_sentence,y1)
print clf1.grid_scores_

#stores the dictionary
pickle.dump(dic,open("relation_terms","wb"))



