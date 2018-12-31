"""*********************************************************

NAME:     ex6

AUTHOR:   Paul Haddon Sanders IV, Ph.D

VERSION:  2. Spam Classification

*********************************************************"""

import re
import numpy as np
from scipy import io
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pdb import set_trace as pb

### Get data ###############################################

mat1  = io.loadmat('spamTrain.mat')
mat2  = io.loadmat('spamTest.mat')
X     = mat1['X']
y     = mat1['y'].ravel()
Xtest = mat2['Xtest']
ytest = mat2['ytest'].ravel()

voca_file = np.loadtxt('vocab.txt',dtype=[('key',np.int16),('word',np.unicode_,16)])

### 2.1 Preprocessing Emails ###############################

def process(file):

    word_indices = []

    with open(file,'r') as f:
        out = "".join(f.readlines())
        out = np.array(out.split(),dtype=object)

    word_indices = []

    for i,val in enumerate(out):

        word_indices.append(voca_file[val == voca_file['word']]['key'])

    word_indices = np.array([item for sublist in word_indices for item in sublist],dtype=int)

    return word_indices

### 2.1.1 Vocabulary List ##################################

mat2 = process('processemailSample1.txt')

### 2.2 Extracting Features from Emails ####################

def email_to_features(word_indices):

    words = np.unique(word_indices)
    
    features = np.zeros(voca_file['key'].shape[0])

    for i,val in enumerate(words):

        features += np.where(val == voca_file['key'],1,0)

    return features

### 2.3 Training SVM for Spam Classification ###############

lr = SVC(kernel='linear',C=0.1)
lr.fit(X,y)

predict_train  = round((lr.predict(X) == y).sum() / y.shape[0],2)
predict_test = round((lr.predict(Xtest) == ytest).sum() / ytest.shape[0],2)

### 2.4 Top Predictors for Spam ############################

indices = lr.coef_.argsort()[0,-15:]
top_pre = voca_file['word'][indices]

### 2.5 Try your own emails ################################

text_file_2 = process('emailSample1.txt')
text_file_3 = process('emailSample2.txt')
text_file_4 = process('spamSample1.txt')
text_file_5 = process('spamSample2.txt')

text_file_2 = email_to_features(text_file_2)
text_file_3 = email_to_features(text_file_3)
text_file_4 = email_to_features(text_file_4)
text_file_5 = email_to_features(text_file_5)

results = lr.predict([text_file_2,text_file_3,text_file_4,text_file_5])

### output #################################################

print("   ex6 : 2. Spam Classification")
print("   Training accuracy = {} %".format(predict_train * 100))
print("   Test accuracy = {} %".format(predict_test * 100))
print("   Top predictors for spam : {}".format(" ".join(top_pre)))
print("   Predicted results : Regular email 1 : {0}".format(results[0]))
print("   Predicted results : Regular email 2 : {0}".format(results[1]))
print("   Predicted results : Spam email 3 : {0}".format(results[2]))
print("   Predicted results : Spam email 4 : {0}".format(results[3]))

############################################################
