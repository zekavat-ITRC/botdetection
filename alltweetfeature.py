from __future__ import unicode_literals

import operator
import timeit

import joblib
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif, SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from NLP import TweeterNormalizer
import numpy as np
from NLP import Tokenizer

#from ClassifyTweets import findSpecialWords, preprocessingandSpecial
from sklearn.svm import LinearSVC

from loadWord2Vec import loadModel, make_agg_vec
from utility import preprocessingandSpecial, findSpecialWords, preprocessing, classifierModel

#read class labels file..............
with open('data\\target.txt') as f:
    Y = f.read().splitlines()
start = timeit.default_timer()
# X is list of tweets
#X=preprocessing('data\\AllTweetsPerUser.xlsx')
bot,human=findSpecialWords(90,250)
#joblib.dump(bot,"botword.pkl")
#joblib.dump(human,"humanword.pkl")
#b=joblib.load("botword.pkl")
#h=joblib.load("humanword.pkl")
tf,special, hashtag,embed=preprocessingandSpecial(bot,human,'data\\AllTweetsPerUser.xlsx')
X=special #np.concatenate((special,embed),axis=1)
#read user account information

df_account=pd.read_excel('data\\AllTweetUser.xlsx',sheet_name='Sheet1')
df_account=df_account.drop(columns="is_bot")
df_account=df_account.drop(columns="1")
#df_account=df_account.drop(columns="9_join")
df_account=df_account.drop(columns="6_join")
df_account=df_account.drop(columns="follow")
df_account=df_account.drop(columns="following")
df_account=df_account.drop(columns="tweets")
df_account=df_account.drop(columns="likes")
df_account=df_account.drop(columns="Age")
df_account=df_account.drop(columns="follow/following")
df_account=df_account.drop(columns="follow/age")
df_account=df_account.drop(columns="following/age")
df_account=df_account.drop(columns="tweet/age")
df_account=df_account.drop(columns="likes/age")
df_account=df_account.drop(columns="verified")


df_account.to_csv ('data\\AllTweet.csv', index = None, header=None, sep=",")



#X_account is account feature vector....................................

X_account = pd.read_csv('data\\AllTweet.csv', sep=",", header=None)


dfX=pd.DataFrame( X,index=[i for i in range(X.shape[0])],columns=['f'+str(i) for i in range(X.shape[1])])
#concat tweet vector and account vector....................................
result_X = pd.concat([dfX, X_account], axis=1)
#########################################################################
#########################################################################
#########################################################################
print(result_X.shape)

result_X.to_csv ('data\\AllFeature-Tweet.csv', index = None,header=True, sep=",")
stop=timeit.default_timer()
print('preprocessing time:' , stop-start)

start=timeit.default_timer()
#split data into train and test ........................................
result_X_train, result_X_test, y_result_train, y_result_test = train_test_split(result_X, Y, test_size=0.20, random_state=200)
# resampling train data to overcome data imbalancing with SMOTE algorithm.....................
#sm = SMOTE(random_state=42)
#X_res, y_res = sm.fit_resample(result_X_train, y_result_train)


#########################################################################
#########################################################################
########select classifier: RandomForest, SVM or Logistic Regression######
#########################################################################

classifier = RandomForestClassifier(n_estimators=400, random_state=0,criterion='gini',min_samples_split=3)
#classifier=svm.SVC(kernel='rbf', gamma=0.4, C=1)
#classifier=LinearSVC(random_state=0, tol=1e-5)
#classifier = LogisticRegression(random_state=0, solver='newton-cg', class_weight='balanced', max_iter=200)

classifierModel(result_X_train, result_X_test, y_result_train, y_result_test,classifier,'model\\allTweetfeature_classification_model.pkl')
#print("==================with resampling======================")
#classifierModel(X_res, result_X_test, y_res, y_result_test,classifier,'model\\allTweetfeature_classification_model.pkl')
stop=timeit.default_timer()
print('classification time:' , stop-start)


print("##############################################################")
print("########Cross-validation#####################")

from sklearn.preprocessing import label_binarize
encoded_column_vector = label_binarize(Y, classes=['0','1']) # ham will be 0 and spam will be 1
encoded_labels = np.ravel(encoded_column_vector) # Reshape array
#######################################################################
#############Cross-validation without Resampling#######################
#######################################################################

scoring_list = ['accuracy','precision', 'recall', 'f1']
scoress = cross_validate(classifier,result_X,encoded_labels,cv=5,scoring=scoring_list,return_train_score=False)
#print(scoress)
print("Accuracy: %0.4f (+/- %0.2f)" % (scoress['test_accuracy'].mean(), scoress['test_accuracy'].std() * 2))
print("Precision: %0.4f (+/- %0.2f)" % (scoress['test_precision'].mean(), scoress['test_precision'].std() * 2))
print("Recall: %0.4f (+/- %0.2f)" % (scoress['test_recall'].mean(), scoress['test_recall'].std() * 2))
print("F-measure: %0.4f (+/- %0.2f)" % (scoress['test_f1'].mean(), scoress['test_f1'].std() * 2))


y_pred = cross_val_predict(classifier,result_X,Y,cv=5)
#print(y_pred)
print("confusion_matrix:")
print(confusion_matrix(Y, y_pred))
print("classification_report:")
print(classification_report(Y, y_pred))
#print("Accuracy: %0.2f (+/- %0.2f)" % accuracy_score(Y, y_pred))

#save class labels in prediction.txt file............................
with open('prediction.txt', 'wb') as f:
    np.savetxt(f, y_pred, fmt='%s')

