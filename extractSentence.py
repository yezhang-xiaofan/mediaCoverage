__author__ = 'zhangye'
import xlrd
import re
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
def convert(str1):
    if(type(str1) is unicode):
        str1 = str1.encode('ascii','ignore')
        str1 = str1.strip('.')
        return re.split(r'\.\. \.\.|,',str1)

sentences = []
y = []
threshold = 3
for file_name in os.listdir("1. excel files"):
    if file_name.endswith(".xls"):
        #print file_name
        book = xlrd.open_workbook("1. excel files/"+file_name)
        first_sheet = book.sheet_by_index(0)
        relation = first_sheet.cell(124,5).value
        code =  first_sheet.cell(125,5).value
        if(code==0.0):
            print file_name
        terms = first_sheet.cell(129,5).value
        if(terms!=-9):
            terms = convert(terms)
        filename = file_name.split('.')[0]
        f = open("5. Press releases/"+filename[:-2]+".txt",'r')
        for line in f:
            #print line
            if(line.startswith("Posted on") or line.startswith("Word Count") or line.startswith("Sentence Count")):
                continue
            sentences.append(line)
            if(code<=threshold):
                y.append(0)
                continue
            if relation.encode('ascii','ignore') in line:
                y.append(1)
            elif(terms!=-9):
                flag = 0
                for i in terms:
                    if i in line:
                        y.append(1)
                        flag = 1
                        #print line
                        break
                if(flag==0):
                    y.append(0)
            else:
                y.append(0)
vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english",
                                    min_df=2,
                                    token_pattern=r"(?u)[a-zA-Z0-9-_/*][a-zA-Z0-9-_/*]+\b",
                                    binary=False, max_features=50000)
sentences = vectorizer.fit_transform(sentences)
lr = LogisticRegression(penalty="l2", fit_intercept=True)
parameters = {"C":[10,1, .1, .01, .001]}
clf0 = GridSearchCV(lr, parameters, scoring='roc_auc')
print "fitting model..."
clf0.fit(sentences,y)
clf0.grid_scores_


