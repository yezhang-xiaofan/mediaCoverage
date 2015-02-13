import numpy as np
import itertools
from sklearn import svm
from scipy.sparse import  coo_matrix
from scipy.sparse import vstack
def rankSVM(X_train,y_train,b_train):
    clf = svm.SVC(kernel='linear', C=.1)
    clf.fit(Xp, yp)
    coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)

    return clf,coef
