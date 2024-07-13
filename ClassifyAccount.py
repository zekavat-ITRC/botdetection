from __future__ import unicode_literals

import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

#read class labels file..............
from sklearn.svm import LinearSVC

from utility import classifierModel
import timeit

start = timeit.default_timer()
with open('data\\target.txt') as f:
    Y = f.read().splitlines()


#read user account information
df_account=pd.read_excel('data\\AllTweetUser.xlsx',sheet_name='Sheet1')

df_account=df_account.drop(columns="is_bot")
df_account=df_account.drop(columns="1")
#remove tweet info===========================================
df_account=df_account.drop(columns="hasmention_sum")
df_account=df_account.drop(columns="1_count")
df_account=df_account.drop(columns="3_sum")
df_account=df_account.drop(columns="4_sum")
df_account=df_account.drop(columns="5_sum")
df_account=df_account.drop(columns="6_join")
df_account=df_account.drop(columns="idEidConv_sum")
df_account=df_account.drop(columns="length_std")
df_account=df_account.drop(columns="length_mean")
df_account=df_account.drop(columns="richness")
df_account=df_account.drop(columns="distinct_word")




df_account.to_csv ('data\\users.csv', index = None, header=None, sep=",")

#X_account is account feature vector....................................
stop = timeit.default_timer()
print('preprocessing time:' , stop-start)

X_account = pd.read_csv('data\\users.csv', sep=",", header=None)

start= timeit.default_timer()
X_account_train, X_account_test, y_account_train, y_account_test = train_test_split(X_account, Y, test_size=0.2, random_state=0)


#########################################################################
#########################################################################
########select classifier: RandomForest, SVM or Logistic Regression######
#########################################################################


classifier = RandomForestClassifier(n_estimators=400, random_state=0,criterion='gini',min_samples_split=3)
#classifier=svm.SVC(kernel='linear', gamma=0.4, C=1)
#classifier=LinearSVC(random_state=0, tol=1e-5)
#classifier = LogisticRegression(random_state=0, solver='newton-cg', class_weight='balanced', max_iter=200)


classifierModel(X_account_train, X_account_test, y_account_train, y_account_test,classifier,'model\\account_classification_model.pkl')
stop = timeit.default_timer()
print('classification time:' , stop-start)



print("##############################################################")
print("########Cross-validation #####################")

from sklearn.preprocessing import label_binarize
encoded_column_vector = label_binarize(Y, classes=['0','1']) # ham will be 0 and spam will be 1
encoded_labels = np.ravel(encoded_column_vector) # Reshape array
#######################################################################
#############Cross-validation without Resampling#######################
#######################################################################

scoring_list = ['accuracy','precision', 'recall', 'f1']
scoress = cross_validate(classifier,X_account,encoded_labels,cv=5,scoring=scoring_list,return_train_score=False)
#print(scoress)
print("Accuracy: %0.4f (+/- %0.2f)" % (scoress['test_accuracy'].mean(), scoress['test_accuracy'].std() * 2))
print("Precision: %0.4f (+/- %0.2f)" % (scoress['test_precision'].mean(), scoress['test_precision'].std() * 2))
print("Recall: %0.4f (+/- %0.2f)" % (scoress['test_recall'].mean(), scoress['test_recall'].std() * 2))
print("F-measure: %0.4f (+/- %0.2f)" % (scoress['test_f1'].mean(), scoress['test_f1'].std() * 2))


y_pred = cross_val_predict(classifier,X_account,Y,cv=5)
#print(y_pred)
print("confusion_matrix:")
print(confusion_matrix(Y, y_pred))
print("classification_report:")
print(classification_report(Y, y_pred))
#print("Accuracy: %0.2f (+/- %0.2f)" % accuracy_score(Y, y_pred))

#save class labels in prediction.txt file............................
with open('prediction.txt', 'wb') as f:
    np.savetxt(f, y_pred, fmt='%s')

