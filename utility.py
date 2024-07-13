from __future__ import unicode_literals

import operator

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
from NLP import TweeterNormalizer,Tokenizer
import numpy as np

from loadWord2Vec import loadModel, make_agg_vec


def preprocessing(filename):
    tweet= TweeterNormalizer(
  #  clear_links_needed="convert",
        clear_punctuation_needed="all"
    )
    # read all tweets for each user
    import codecs
    #with codecs.open(filename, 'r', encoding='utf-8',errors='ignore') as fdata:

    df_tweets=pd.read_excel(filename,sheet_name='Sheet1')
    Row_list_Embed = []
    Row_list = []
    tkn = Tokenizer()
    mymodel, featuresize, wordSet = loadModel()
    for index, row in df_tweets.iterrows():
        normal_tweet = tweet.normalize(row["6"])
        avg = make_agg_vec(tkn.word_tokenizer(normal_tweet), mymodel, featuresize, wordSet)
        #        print(avg)
        Row_list_Embed.append(avg)
        Row_list.append(normal_tweet)

    tfidfconverter = TfidfVectorizer(max_features=1000, min_df=0.1, max_df=0.7, ngram_range=(1, 3))
    X = tfidfconverter.fit_transform(Row_list).toarray()
    joblib.dump(tfidfconverter,"tfidf_model2.pkl")
    tfidfconverter.get_feature_names()
    # res = dict(zip(tfidfconverter.get_feature_names(),
    #                mutual_info_classif(X, Y, discrete_features=True)
    #                ))
    # # sort TF-IDF features .............
    # sorted_x = sorted(res.items(), key=operator.itemgetter(1))
    #   print(sorted_x)
    listC = np.concatenate([X, Row_list_Embed], axis=1).tolist()

    # listC = []
    return X

def preprocessingandSpecial(botword,humanword, tweetFile):
    mymodel, featuresize, wordSet = loadModel()

    tweet= TweeterNormalizer(
  #  clear_links_needed="convert",
        clear_punctuation_needed="all"
    )
    # read all tweets for each user
    if str(tweetFile).endswith(".xlsx"):
        df_tweets=pd.read_excel(tweetFile,sheet_name='Sheet1')
    elif str(tweetFile).endswith(".csv"):
        df_tweets = pd.read_csv(tweetFile, sep=',', encoding='utf-8')

    Row_list = []
    botvalue=[]
    humanvalue=[]
    Row_list_Embed = []
    hashtag = []
    for index, row in df_tweets.iterrows():
        botword = dict.fromkeys(botword, 0)
        humanword = dict.fromkeys(humanword, 0)
        if str(tweetFile).endswith(".xlsx"):
            normal_tweet = tweet.normalize(row["6"])
        elif str(tweetFile).endswith(".csv"):
            normal_tweet = tweet.normalize(row[2])

        avg = make_agg_vec(Tokenizer().word_tokenizer(normal_tweet), mymodel, featuresize, wordSet)
        Row_list_Embed.append(avg)
        #print(index)

        count = 0
        tokensTweet = Tokenizer().word_tokenizer(normal_tweet)
        for t in tokensTweet:
            if str(t).startswith("#"):
                count=count+1
            if t in botword:
                botword[t]=botword[t]+1
            if t in humanword:
                humanword[t]=humanword[t]+1
        hashtag.append(count)
        botvalue.append(list(botword.values()))
        humanvalue.append(list(humanword.values()))
        Row_list.append(normal_tweet)


    tfidfconverter = TfidfVectorizer(max_features=400,min_df=0.1, max_df=0.7 ,ngram_range=(1, 3))
    X = tfidfconverter.fit_transform(Row_list).toarray()

    special= [x+y for x,y in zip(botvalue,humanvalue)]
    a = np.array(X)
    b= np.array(special)
    d=np.array(hashtag)
    d2 = np.reshape(d, (-1, 1))
    #print(b.shape)
    #print(d2.shape)
    all= np.concatenate((d2 , b), axis=1)
    all2=np.concatenate((b,a),axis=1)
    all3 = np.concatenate([b, Row_list_Embed], axis=1)


    return a,b,d2,Row_list_Embed

def insertToDict(normal_tweet , dict_w , tkn):
    tokensTweet = tkn.word_tokenizer(normal_tweet)
    for r in tokensTweet:
        if r in dict_w:
            dict_w[r] = dict_w[r] + 1
        else:
            dict_w[r] = 1
    return dict_w

def findSpecialWord(class1,class2, threshold):
    c=dict()

    for key, value in class1.items():
        freqB=value
        freqH=class2.get(key)
        if freqH:
            fb=(freqH+freqB)/freqB
           # fh=(freqH+freqB)/freqH
            diff=freqB-freqH
            if diff > threshold:
                c[key]=fb
        elif freqB > threshold:
            c[key]=1
    return c

def findSpecialWords(threshold, most):

    tweet= TweeterNormalizer(
  #  clear_links_needed="convert",
        clear_stopwords_needed=True,
        clear_number_needed=True,
        clear_links_needed='convert',
        clear_punctuation_needed="all"
    )
    # read all tweets for each user
    df_tweets=pd.read_excel('data\\AllTweetsPerUser.xlsx',sheet_name='Sheet1')
    tkn = Tokenizer()
    botWord=dict()
    humanWord = dict()
    for index, row in df_tweets.iterrows():
        if row["is_bot"]==1:
            normal_tweet= tweet.normalize(row["6"])
            insertToDict(normal_tweet,botWord,tkn)
        else:
            normal_tweet = tweet.normalize(row["6"])
            insertToDict(normal_tweet, humanWord, tkn)
    cb =findSpecialWord(botWord,humanWord,threshold)
    ch = findSpecialWord(humanWord,botWord,threshold)

    from collections import Counter
    Counterb = Counter(cb)
    most_occurb = Counterb.most_common(most)
    listb, list2 = zip(*most_occurb)
    #print(most_occurb)
    #print(len(most_occurb))
    Counterh = Counter(ch)
    most_occurh = Counterh.most_common(most)
    #print(most_occurh)
    #print(len(most_occurh))
    listh, list2 = zip(*most_occurh)
    #listh= map(list, zip(*most_occurh))

    #botwordList = dict.fromkeys(listb, 0)
    #humanwordList = dict.fromkeys(listh, 0)

   # return botwordList,humanwordList
    return listb,listh
  #  joblib.dump(tfidfconverter,"tf-idf-tweet-model.pkl")



def getAccuracy(pre,ytest):
    count = 0
    for i in range(len(ytest)):
        if ytest[i]==pre[i]:
            count+=1
            acc = float(count)/len(ytest)
    return acc


def classifierModel(X_train, X_test, y_train, y_test,classifier, modelfile):
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, modelfile)
   # clf = joblib.load('classifyAccount_model.pkl')

    y_pred = classifier.predict(X_test)

    print("Confusion matrix : ")
    print(confusion_matrix(y_test,y_pred))
    print("Classification_report : ")
    print(classification_report(y_test,y_pred))
    print("Accuracy %0.4f :" % accuracy_score(y_test, y_pred))

    return y_pred
