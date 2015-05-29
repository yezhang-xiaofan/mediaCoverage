'''
This module implements the actual model building 
and exploration bit for factiva. All data scraping
is done elsewhere (factiva.py).
'''
import random
import csv 
import pdb
import os 
import operator 

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

import numpy as np 
from numpy.linalg import norm
from textblob import TextBlob
import re

import pylab
import seaborn as sns

def _get_ranked_features(clf, vectorizer, n=50):
    c_f = sorted(zip(clf.best_estimator_.raw_coef_[0], vectorizer.get_feature_names()))
    #if n == 0:
    #    n = len(c_f)/2

    #top = zip(c_f[:n], c_f[:-(n+1):-1])
    weights_d = dict(zip(vectorizer.get_feature_names(), clf.best_estimator_.raw_coef_[0]))
    return c_f, weights_d 
    #return top 

def feature_weight_comparison():
    '''
    a simple explorative, contrasting analysis: get top 100 features for both
    models (max 200 in all, assuming no overlap) and then sort by absolute 
    differences between them
    '''
    # construct shared feature-space using both datasets
    v = get_combined_vectorizer()
    jama_clf = perform_classification(jama=True, vectorizer=v)
    
    ranked_jama_features, jama_feature_weight_d = _get_ranked_features(jama_clf, v)

    factiva_clf = perform_classification(jama=False, vectorizer=v)
    ranked_factiva_features, factiva_feature_weight_d = _get_ranked_features(factiva_clf, v)

    factiva_norm = norm(factiva_feature_weight_d.values())
    jama_norm = norm(jama_feature_weight_d.values())
    
    factiva_jama, factiva_jama_abs = {}, {}
    for tuple_ in ranked_factiva_features[-20:]:
        factiva_val, f = tuple_
        try:
            jama_val = jama_feature_weight_d[f]
        except:
            pdb.set_trace()

        #factiva_jama[f] = (factiva_val, jama_val)
        factiva_jama[f] = factiva_val/factiva_norm - jama_val/jama_norm
        factiva_jama_abs[f] = abs(factiva_val/factiva_norm - jama_val/jama_norm)
    '''
    jama_factiva = {}
    for f, factiva_val in ranked_factiva_features[-20:]:
        jama_val = jama_feature_weight_d[f]
        factiva_jama[f] = (factiva_val, jama_val)
    '''
    # sorted(map(abs, lista), reverse=True)
    #pdb.set_trace()
    sorted_x = sorted(factiva_jama.items(), key=operator.itemgetter(1), reverse=True)
    #pdb.set_trace()

    out_str = [
        r'''\begin{table} 
            \begin{tabular}{l | c } 
            term & \emph{Reuters} - \emph{JAMA} weight \\
            \hline
        ''']

    low = sorted_x[:10]
    for f, v in low:
        out_str.append("%s & %s \\\\" % (f, "%.2f" % v))

    out_str.append("\hline")

    high = sorted_x[-10:]
    for f, v in high:
        out_str.append("%s & %s \\\\" % (f, "-%.2f" % v))

    out_str.append(r'''\end{tabular} 
                    \caption{Features with the largest magnitude of difference (with respect to their normalized estimated coefficients) between the Reuters and JAMA datasets. } 
                    \end{table}''')
    print "\n".join(out_str)
    pdb.set_trace()
### so, e.g.,
# > v = factiva_model.get_combined_vectorizer()
# > jama_clf = factiva_model.perform_classification(jama=True, vectorizer=v)
# > factiva_clf = factiva_model.perform_classification(jama=False, vectorizer=v)
####
def get_combined_vectorizer(max_features=50000):
    texts, jama_y, of_interest = load_Xy(jama=True)
    factiva_texts, factiva_y, of_interest = load_Xy(jama=False)
  
    texts = list(texts)
    texts.extend(factiva_texts)

    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", 
                                    min_df=100, token_pattern=r"(?u)[a-zA-Z0-9-_/*][a-zA-Z0-9-_/*]+\b",
                                    binary=False, max_features=max_features)
    vectorizer.fit(texts)
    #pdb.set_trace()
    return vectorizer
 
def perform_classification(max_features=50000, jama=False, vectorizer=None):
    texts, y, of_interest = load_Xy(jama=jama) #process_files()

    y = np.array(y)
    if vectorizer is None:
        vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", 
                                        min_df=100, token_pattern=r"(?u)[a-zA-Z0-9-_/*][a-zA-Z0-9-_/*]+\b",
                                        binary=False, max_features=max_features)
        vectorizer.fit(texts)

    #X = vectorizer.fit_transform(texts)
    X = vectorizer.transform(texts)
    

    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    
    print "total training examples: %s; num positive: %s" % (X.shape[0], y[y>0].shape[0])
    #lr = LogisticRegression(class_weight="auto", penalty="l2", fit_intercept=True)
    lr = LogisticRegression(penalty="l2", fit_intercept=True)

    #.1,.01, 
    parameters = {"C":[.1, .01, .001]}
    #parameters = {"C":[1]}
    #parameters = {"C":[100000]} # basically, no regularization
    clf0 = GridSearchCV(lr, parameters, scoring='f1')
    print "fitting model..."
    clf0.fit(X,y)
    print "done."
    
    #show_most_informative_features(vectorizer, clf)
    if not jama:
        caption_str = '''
            \caption{Top fifty features and associated weights for \emph{Reuters} corpus, ranked by magnitude.}
        '''
    else:
        caption_str = '''
            \caption{Top fifty features and associated weights for the \emph{JAMA} corpus, ranked by magnitude.}
        '''        

    print texify_most_informative_features(vectorizer, clf0, caption_str)
    
    # assess performance, too
    #clf = GridSearchCV(lr, parameters, scoring='f1')

    kf = cross_validation.KFold(X.shape[0], shuffle="true", n_folds=10)
    fs, aucs = [],[]
    fold = 0
    for train, test in kf:
        clf = GridSearchCV(lr, parameters, scoring='f1')
        clf.fit(X[train], y[train])
   
        probs = clf.predict_proba(X[test])
        
        #aucs.append(sklearn.metrics.roc_auc_score(y[test], probs))
        cur_auc = sklearn.metrics.roc_auc_score(y[test], probs[:,1])
        aucs.append(cur_auc)
        preds = clf.predict(X[test])
        fs.append(sklearn.metrics.f1_score(y[test], preds))

        if fold == 0:
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y[test], probs[:,1])
            pylab.clf()

            fout = "roc_"
            if jama:
                fout = fout + "jama.pdf"
            else:
                fout = fout + "factiva.pdf"

            pylab.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % cur_auc)
            pylab.plot([0,1], [0,1], 'k--')
            pylab.xlim((-0.025,1.025))
            pylab.ylim((-0.025,1.025))
            pylab.xlabel("false positive rate")
            pylab.ylabel("true positive rate")
            pylab.title("ROC curve (area = %0.2f)" % cur_auc)
            pylab.tight_layout()
            pylab.savefig(fout)


        fold += 1


    #print sklearn.cross_validation.cross_val_score(clf, X, y, scoring='f1', cv=5, shuffle="true" )
    #pdb.set_trace()
    print "average auc: %s" % (sum(aucs)/float(len(aucs)))
    print "average fs: %s" % (sum(fs)/float(len(fs)))
    #print "ABOUT TO RETURN"
    pdb.set_trace()
    return clf0

def combine_files(start=0, end=30, 
        info_out="all_factive_article_info.csv", 
        matched_out="all_factiva_matched_articles.csv"):
    combined_info = []
    combined_matched = []
    for factiva_file_num in range(start, end):
        try:
            info_path="Factiva%s_article_info_redux.csv" % factiva_file_num
            matched_sample_path = "Factiva%s_matched_articles.csv" % factiva_file_num
            combined_info.extend(open(info_path).readlines())
            combined_matched.extend(open(matched_sample_path).readlines())
        except:
            print "failed to parse factiva file: %s" % factiva_file_num

    with open(info_out, 'wb') as outf:
        outf.write(''.join(combined_info))

    with open(matched_out, 'wb') as outf:
        outf.write(''.join(combined_matched))
    print "ok!"
            
def filter_matched_articles():
    # we may have grabbed articles that *did* have a press release 
    # during our matched sampling -- get rid of them here.
    factiva_pmids = [x[0] for x in list(csv.reader(open("all_factive_article_info.csv")))]

    filtered = []
    with open("all_factiva_matched_articles.csv") as matched_f:
        matched = list(csv.reader(matched_f))
        print "%s matched articles (unfiltered)" % len(matched)
        for l in matched:
            if not l[1] in factiva_pmids:
                filtered.append(l)

    print "filtered articles: %s" % len(filtered)
    with open("all_factiva_matched_articles_filtered.csv", 'w')  as outf:
        csv_writer = csv.writer(outf)
        for l in filtered:
            csv_writer.writerow(l)

###
# due to our refactoring, this method is a bit redundant,
# since it's just calling out to process_file, below...
def load_Xy(dir_="_files_for_analysis", jama=False):
    if not jama:
        article_info_path = os.path.join(dir_, "all_factive_article_info.csv")
        matched_article_path = os.path.join(dir_, "all_factiva_matched_articles_filtered.csv")
        
    else:
        dir_ = "../jama/_files_for_analysis"
        article_info_path = os.path.join(dir_, "jama_article_info.csv")
        matched_article_path = os.path.join(dir_, "jama_pmids.txt_matched_articles_filtered.csv")
        
    X, y, of_interest = process_file(article_info_path, matched_article_path)

    return X, y, of_interest

'''
!!!
antiquated: instead, use combine_files to generate 
a single file, then use load_Xy
'''
def process_files(start=0, end=30):
    X, y = [], []
    of_interest = [] # abtracts/titles with tokens we're curious about
    for factiva_num in range(start,end):
        try:
            print "on file %s" % factiva_num
            cur_X, cur_y, cur_of_interest = process_file(factiva_num)
            X.extend(cur_X)
            y.extend(cur_y)
            of_interest.extend(cur_of_interest)
        except:
            print "failed to parse factiva file: %s" % factiva_num

    return X,y, of_interest

#def process_file(factiva_file_num=1):
#    info_path="Factiva%s_article_info_redux.csv" % factiva_file_num
#    matched_sample_path = "Factiva%s_matched_articles.csv" % factiva_file_num
def process_file(info_path, matched_sample_path, jama=False):

    def prepend_to_words(text, prepend_str="TI-"):
        prepended_words = []

        return " ".join(
                [prepend_str + text for text in text.split(" ") if not 
                        text.strip() in ("missing", "", "&")])

    def clean_up_mesh(mesh):
        mesh_tokens = mesh.split(" ")
        clean_tokens = []
        remove_these = ["&", "(", ")"]
        for t in mesh_tokens:
            for to_remove in remove_these:
                t = t.replace(to_remove, "")

            if t.startswith("*"):
                t = t[1:]

            if len(t) > 0:
                clean_tokens.append(t)
        return " ".join(clean_tokens)


    # right now, we're just merging all the articles
    # together; we may want to move to a more 'pairwise'
    # approach, where the model is explicitly defined 
    # to make pairwise decisions regarding whether or not 
    # one of two articles will get picked up.
    X, y = [], []
    of_interest = []
    len_title = 0
    len_abstract = 0
    with open(info_path, 'rU') as input_file:
        reader = csv.reader(input_file)
        for line in reader:
            PMID, title, journal, \
                     authors, affiliation, abstract, MESH = line
            print PMID
            MESH = clean_up_mesh(MESH)

            # append title indicators to tokens in titles.
            # @TODO technically, should probably stopword
            # the titles before doing this...

            ### these are missing entries; before I was including
            # these in the data, which led to "missing" being
            # an ostensibly predictive feature...
            if title.strip() == "Missing":
                title = ""

            if abstract.strip() == "Missing":
                abstract = ""

            if affiliation.strip() == "Missing":
                affiliation = ""

            if MESH.strip() == "Missing":
                MESH = ""

            #if "2012" in abstract.split(" "):
            tokens = abstract.split(" ")
            if "patients" in tokens:
               of_interest.append((abstract, 1))
        

            if not abstract == "":
                len_title += len(TextBlob(title).words)
                len_abstract += len(TextBlob(abstract).words)
                all_features = [#prepend_to_words(journal, prepend_str="JOURNAL-"),
                                prepend_to_words(title, prepend_str="TI-"),
                                #remove "IMPORTANCE:"
                                abstract,
                                #prepend_to_words(affiliation, prepend_str="AF-"),
                                prepend_to_words(MESH, prepend_str="MH-")]
                
                #if "Numerical" in MESH or "numerical" in MESH:
              
                X.append(" ".join(all_features))
                y.append(1)
                #if "MHxxx" in prepend_to_words(MESH, prepend_str="MHxxx").split(" "):

    ###
    # now grab the matched articles from a file that 
    # is named, by convention, FactivaN_matched_articles.csv
    ###
    with open(matched_sample_path, 'rU') as matched_input_file:
        reader = csv.reader(matched_input_file)
        for line in reader:
            source_pmid, matched_pmid, title, journal, \
                 authors, affiliation, abstract, MESH = line

            MESH = clean_up_mesh(MESH)

            if title.strip() == "Missing":
                title = ""

            if abstract.strip() == "Missing":
                abstract = ""

            if affiliation.strip() == "Missing":
                affiliation = ""

            if MESH.strip() == "Missing":
                MESH = ""


            if not abstract == "":
                len_title += len(TextBlob(title).words)
                len_abstract += len(TextBlob(abstract).words)
                X.append(" ".join([#prepend_to_words(journal, prepend_str="JOURNAL-"),
                                   prepend_to_words(title, prepend_str="TI-"), 
                                   abstract,
                                   #prepend_to_words(affiliation, prepend_str="AF-"),
                                   prepend_to_words(MESH, prepend_str="MH-")]))
                y.append(-1)

                tokens = abstract.split(" ")

                ### WHY is patients so over-represented in negative instances??
                if "patients" in tokens:
                    of_interest.append((abstract, -1))
    print "average length of title is: " + str(len_title/len(y))
    print "average length of abstract is: " + str(len_abstract/len(y))
    # shuffle it so that we don't get all the inclues (1's) at
    # the front of the list
    training = zip(X,y)
    random.shuffle(training)
    X,y = zip(*training)
   
    return X,y,of_interest

def construct_training_data_for_article(row):
    pass 



def texify_most_informative_features(vectorizer, clf, caption, n=50):
    ###
    # note that in the multi-class case, clf.coef_ will
    # have k weight vectors, which I believe are one per
    # each class (i.e., each is a classifier discriminating
    # one class versus the rest). 
    #c_f = sorted(zip(clf.coef_[2], vectorizer.get_feature_names()))
    c_f = sorted(zip(clf.best_estimator_.raw_coef_[0], vectorizer.get_feature_names()))
    if n == 0:
        n = len(c_f)/2

    top = zip(c_f[:n], c_f[:-(n+1):-1])
    print
    print "%d most informative features:" % (n, )
    out_str = [
        r'''\begin{table} 
            \begin{tabular}{l c | l c}

        '''
    ]
    out_str.append(r"\multicolumn{2}{c}{\emph{negative}} & \multicolumn{2}{c}{\emph{positive}} \\")
    for (c1, f1), (c2, f2) in top:
        #out_str.append("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))
        out_str.append("%.3f & %s & %.3f & %s \\\\" % (c1, f1, c2, f2))

    #
    out_str.append(r"\end{tabular}")
    out_str.append("%s" % caption)
    out_str.append(r"\end{table}")

    feature_str = "\n".join(out_str)
    
    print "\n"
    print feature_str
    #return (feature_str, top)

def show_most_informative_features(vectorizer, clf, n=50):
    ###
    # note that in the multi-class case, clf.coef_ will
    # have k weight vectors, which I believe are one per
    # each class (i.e., each is a classifier discriminating
    # one class versus the rest). 
    #c_f = sorted(zip(clf.coef_[2], vectorizer.get_feature_names()))
    c_f = sorted(zip(clf.best_estimator_.raw_coef_[0], vectorizer.get_feature_names()))
    if n == 0:
        n = len(c_f)/2

    top = zip(c_f[:n], c_f[:-(n+1):-1])
    print
    print "%d most informative features:" % (n, )
    out_str = []
    for (c1, f1), (c2, f2) in top:
        out_str.append("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))
    feature_str = "\n".join(out_str)
    print feature_str
    #return (feature_str, top)




