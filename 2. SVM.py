
# coding: utf-8

# In[18]:


#!/usr/bin/python
""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

def svm(features_train, features_test, labels_train, labels_test, limit_train = 'No', kernel='linear', C=1.0):
    # declare classifier
    clf = SVC(kernel = kernel, C=C)
    
    # set training set
    if limit_train == 'Yes':
        features_train = features_train[:len(features_train)/100] 
        labels_train = labels_train[:len(labels_train)/100]
    
    # train classifier
    t0 = time()
    clf.fit(features_train,labels_train)
    print "\ntime to train:", round(time()-t0, 3), "s"
    
    # predict classifer
    pred = clf.predict(features_test)
    print "\ntime to predict:", round(time()-t0, 3), "s"

    accuracy = accuracy_score(pred, labels_test)

    print '\nprediction accuracy = {0}'.format(accuracy)
    return pred

pred = svm(features_train, features_test, labels_train, labels_test, 'No', 'rbf', 10000)

print "Predictions for records 10, 26 and 50, respectively:", pred[10], ",", pred[26],",", pred[50]
print "Number of Chris emails:", sum(pred)

