__author__ = 'zhangye'
#this file predicts relationship and its level in sentence level
import xlrd
import re
import os
from nltk.tokenize import sent_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
import codecs
import numpy as np
import unicodedata
import pickle
from sklearn.metrics import f1_score
from operator import itemgetter
from sklearn import linear_model
def convert(str1):
    if(type(str1) is unicode):
        str1 = str1.encode('ascii','ignore')
        str1 = str1.strip('.')
        str1 = str1.lower()
        temp =  re.split(r'\.\. \.\.|,',str1)
        return [a for a in temp if a.strip().isspace()==False and a]

sentences = []
y = []
threshold = 0
level = []

#write postiive sentences to sen_file
#sen_file = open("sentence.txt","w")
ignore = ["posted on","word count","sentence","title"]
dic = {}
def is_ignore(text):
    for str in ignore:
        if(str in text):
            return 1
    return 0
Chambers_sentence = "Chambers_sen/"
j = 1
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
            #print terms
            terms = convert(terms)
            #sen_file.write(file_name+" "+', '.join(terms)+" \n")
        filename = file_name.split('.')[0]
        f = codecs.open("5. Press releases/"+filename[:-2]+".txt",'r',encoding='utf-8')
        text = f.readlines()
        text = [i.encode('ascii','ignore') for i in text]
        writeFile = open(Chambers_sentence+str(j),'wb')
        #sents = sent_tokenize(text)
        for line in text:
            original = line
            if(line.strip().isspace()):
                continue
            #print line
            line = line.lower().strip()
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
                #sen_file.write(file_name+" "+line+"\n")
            elif(terms!=-9):
                flag = 0
                for i in terms:
                    if(i=="as above" or i.isspace() or not i):
                        continue
                    i = i.strip()
                    if(dic.has_key(i)):
                        dic[i].append(file_name)
                    else:
                        dic[i] = [file_name]
                    if i in line:
                        y.append(1)
                        flag = 1
                        level.append(code)
                        #print "sen "+line
                        #sen_file.write(file_name+" "+line+"\n")
                        #print line
                        break
                if(flag==0):
                    y.append(0)
            else:
                y.append(0)

            #write the current line with the label into the sentences directory
            if line:
                writeFile.write(line+"\t"+str(y[-1])+"\n")
        writeFile.close()
        j+=1

#stores the dictionary
pickle.dump(dic,open("relation_terms","wb"))
parameters = [1000,100,10,1, .1, .01, .001,0.0001,0.00001]
level = np.array(level)
threshold_index = np.nonzero(level>3)
y1 = np.zeros(len(level))
y1[threshold_index] = 1
kf1 = cross_validation.StratifiedKFold(y1,n_folds=5,shuffle=True)
lr1 = LogisticRegression(penalty="l2",fit_intercept=True)
clf1 = GridSearchCV(lr1, parameters, scoring='f1',cv=kf1)
#clf1.fit(relation_sentence,y1)
print clf1.grid_scores_


#sen_file.close()
labeled_sen = []
labeled_y = []

#read the Oxford press release labeled by the dictionary terms
for file_name in os.listdir("PR_Oxford_Sentence"):
    if(not file_name.endswith(".txt")):
        continue
    temp = open("PR_Oxford_Sentence/"+file_name)
    for line in temp:
        line = line.strip()
        label = int(line[-1].strip())
        sen = line[:-1]
        labeled_sen.append(sen.strip())
        labeled_y.append(label)
    temp.close()

#sentences += labeled_sen
#y += labeled_y
vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english",
                                    min_df=2,
                                    #token_pattern=r"(?u)[a-zA-Z0-9-_/*][a-zA-Z0-9-_/*]+\b",
                                    binary=False, max_features=50000)
#sentences = vectorizer.fit_transform(sentences)
y = np.array(y)
#relation_index = np.nonzero(y)
#relation_sentence = sentences[relation_index]
kf = cross_validation.KFold(len(y),n_folds=5,shuffle=True)
lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto')


#labeled_sen = []
#labeled_y = []
for p in parameters:
    #lr = LogisticRegression(penalty="l2", fit_intercept=True,class_weight='auto',C=p)
    lr = linear_model.SGDClassifier(loss='log',penalty='l2',fit_intercept=True,class_weight='auto',alpha=p,n_iter=50)
    mean = []
    for train_index, test_index in kf:
        train_sentence = list(itemgetter(*train_index)(sentences)) + labeled_sen
        train_label = list(itemgetter(*train_index)(y)) + labeled_y
        test_sentence = list(itemgetter(*test_index)(sentences))
        train_sentence = vectorizer.fit_transform(train_sentence)
        test_sentence = vectorizer.transform(test_sentence)
        test_label = list(itemgetter(*test_index)(y))
        ins_weight = np.ones(len(train_index))
        ins_weight = np.concatenate((ins_weight,np.ones(len(labeled_y))*0.3))
        lr.fit(train_sentence,np.array(train_label),sample_weight=ins_weight)
        predict = lr.predict(test_sentence)
        f1 = f1_score(test_label,predict)
        mean.append(f1)
    print(str(p)+" "+str(sum(mean)/len(mean))+"\n")

clf0 = GridSearchCV(lr, parameters, scoring='f1',cv=kf)
print "fitting model..."
clf0.fit(sentences,y)
print clf0.grid_scores_







