from __future__ import unicode_literals

import operator
import timeit

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif, SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from NLP import TweeterNormalizer
import pickle
from sklearn.externals import joblib
from NLP import Tokenizer

from collections import Counter

from loadWord2Vec import loadModel, make_agg_vec
from utility import findSpecialWords, preprocessingandSpecial, classifierModel

#read class labels file..............
with open('data\\target.txt') as f:
    Y = f.read().splitlines()

start=timeit.default_timer()
# X is list of tweets
bot,human=findSpecialWords(90,250)
tf, special, hashtag,embed=preprocessingandSpecial(bot,human,'data\\AllTweetsPerUser.xlsx')
X=np.concatenate((special,embed), axis=1)
#X=preprocessing()
#print(X.shape)
stop=timeit.default_timer()
print('preprocessing time:' , stop-start)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


#########################################################################
#########################################################################
########select classifier: RandomForest, SVM or Logistic Regression######
#########################################################################

start=timeit.default_timer()
classifier = RandomForestClassifier(n_estimators=400, random_state=0,criterion='gini',min_samples_split=3)
#classifier=svm.SVC(kernel='rbf', gamma=0.4, C=1)
#classifier = LogisticRegression(random_state=0, solver='newton-cg', class_weight='balanced', max_iter=200)

classifierModel(X_train, X_test, y_train, y_test,classifier, 'model\\tweet_classification_model.pkl')
stop=timeit.default_timer()
print('classification time:' , stop-start)



print("##############################################################")
print("########Cross-validation without Resampling#####################")

from sklearn.preprocessing import label_binarize
encoded_column_vector = label_binarize(Y, classes=['0','1']) # ham will be 0 and spam will be 1
encoded_labels = np.ravel(encoded_column_vector) # Reshape array
#######################################################################
#############Cross-validation without Resampling#######################
#######################################################################

scoring_list = ['accuracy','precision', 'recall', 'f1']
scoress = cross_validate(classifier,X,encoded_labels,cv=5,scoring=scoring_list,return_train_score=False)
#print(scoress)
print("Accuracy: %0.4f (+/- %0.2f)" % (scoress['test_accuracy'].mean(), scoress['test_accuracy'].std() * 2))
print("Precision: %0.4f (+/- %0.2f)" % (scoress['test_precision'].mean(), scoress['test_precision'].std() * 2))
print("Recall: %0.4f (+/- %0.2f)" % (scoress['test_recall'].mean(), scoress['test_recall'].std() * 2))
print("F-measure: %0.4f (+/- %0.2f)" % (scoress['test_f1'].mean(), scoress['test_f1'].std() * 2))


y_pred = cross_val_predict(classifier,X,Y,cv=5)
#print(y_pred)
print("confusion_matrix:")
print(confusion_matrix(Y, y_pred))
print("classification_report:")
print(classification_report(Y, y_pred))
#print("Accuracy: %0.2f (+/- %0.2f)" % accuracy_score(Y, y_pred))

#save class labels in prediction.txt file............................
with open('prediction.txt', 'wb') as f:
    np.savetxt(f, y_pred, fmt='%s')

