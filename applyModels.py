from builtins import object
from collections import Counter


import joblib
import numpy as np
import pandas as pd

from NLP.src.normalizer.tweeter_normalizer import TweeterNormalizer
from NLP import Tokenizer

from loadWord2Vec import loadModel, make_agg_vec
from utility import preprocessingandSpecial


def loadClassifierModel(X_test , model):
    clf = joblib.load(model)
    y_pred = clf.predict(X_test)
    return y_pred



def applyAccountModel(Testfile, model, outputPredictionFile):
    # X_account is account feature vector....................................

    X_account = pd.read_csv(Testfile, sep=",", header=None)
    X_account=X_account.drop(columns=[0,1,2,3,4,5,6,7,8,9])
    y_pred = loadClassifierModel(X_account,model)

    with open(outputPredictionFile, 'w', encoding="utf-8") as filehandle:
        filehandle.writelines("%s\n" % place for place in y_pred)
    return y_pred

def preprocessing(testFile, tfidfModel):
    tweet= TweeterNormalizer(
  #  clear_links_needed="convert",
        clear_punctuation_needed="all"
    )
    # read all tweets for each user
    df_tweets=pd.read_csv(testFile,encoding='utf-8',sep=",")
    Row_list = []

    for index, row in df_tweets.iterrows():
        normal_tweet= tweet.normalize(row[2])
        Row_list.append(normal_tweet)

    tfidfconverter = joblib.load(tfidfModel)   #TfidfVectorizer(max_features=1000, min_df=0.3, max_df=0.75,ngram_range=(1, 3))
    X = tfidfconverter.transform(Row_list).toarray()
    tfidfconverter.get_feature_names()
    return X

def preprocessingWithEmbedding(filename,tfidf_model):
    tweet= TweeterNormalizer(
  #  clear_links_needed="convert",
        clear_punctuation_needed="all"
    )
    # read all tweets for each user
    df_tweets = pd.read_csv(filename,sep=',',encoding='utf-8')

    Row_list_Embed = []
    Row_list = []
    tkn = Tokenizer()
    mymodel, featuresize, wordSet = loadModel()
    for index, row in df_tweets.iterrows():
       # print(row)
        normal_tweet = tweet.normalize(row[2])
        avg = make_agg_vec(tkn.word_tokenizer(normal_tweet), mymodel, featuresize, wordSet)
        #        print(avg)
        Row_list_Embed.append(avg)
        Row_list.append(normal_tweet)

    #tfidfconverter = TfidfVectorizer(max_features=1000, min_df=0.2, max_df=0.5, ngram_range=(1, 3))
    tfidfconverter= joblib.load(tfidf_model)
    X = tfidfconverter.transform(Row_list).toarray()

    listC = np.concatenate([X, Row_list_Embed], axis=1).tolist()

    # listC = []
    return listC


def applyTweetModel(Testfile, classificationModel, outputPredictionFile,botfile , humanfile):
    bot=joblib.load(botfile)
    human=joblib.load(humanfile)
    tf,special, hashtag,embed = preprocessingandSpecial(bot, human,Testfile)
    X= np.concatenate((special,embed),axis=1)

    #X = preprocessing(Testfile, tfidfModel)

    clf = joblib.load(classificationModel)
    y_pred = clf.predict(X)
    with open(outputPredictionFile, 'w', encoding="utf-8") as filehandle:
        filehandle.writelines("%s\n" % place for place in y_pred)
    return y_pred


# X is list of tweets


def applyAllFeatureModel(Testfile_profile,Testfile_tweet, classificationModel, tfidfModel, outputPredictionFile):
    X = preprocessing(Testfile_tweet, tfidfModel)

    # X_account is account feature vector....................................

    X_account = pd.read_csv(Testfile_profile, sep=",", header=None)

    dfX = pd.DataFrame(X, index=[i for i in range(X.shape[0])], columns=['f' + str(i) for i in range(X.shape[1])])
    #print(pd.isnull(dfX).sum())
    # concat tweet vector and account vector....................................
    result_X = pd.concat([dfX, X_account], axis=1, join='inner')
    #print(pd.isnull(result_X).sum())

    result_X.to_csv('data\\AllFeature.csv', index=None, header=True, sep=",")

    clf = joblib.load(classificationModel)
    y_pred = clf.predict(result_X)
    with open(outputPredictionFile, 'w', encoding="utf-8") as filehandle:
        filehandle.writelines("%s\n" % place for place in y_pred)
    return y_pred

def applyTweetEmbedModel(Testfile, classificationModel, tfidfModel, outputPredictionFile):

    X = preprocessingWithEmbedding(Testfile, tfidfModel)
    #X = preprocessing(Testfile, tfidfModel)
    clf = joblib.load(classificationModel)
    y_pred = clf.predict(X)
    with open(outputPredictionFile, 'w', encoding="utf-8") as filehandle:
        filehandle.writelines("%s\n" % place for place in y_pred)

def applyAllFeatureModel_special(Testfile_profile,Testfile_tweet, classificationModel,botWord,humanWord, outputPredictionFile):
    botWord = joblib.load(botWord)
    humanWord = joblib.load(humanWord)
    tf, special, hashtag, embed = preprocessingandSpecial(botWord, humanWord, Testfile_tweet)
    X = special

    # X_account is account feature vector....................................

    X_account = pd.read_csv(Testfile_profile, sep=",", header=None)

    dfX = pd.DataFrame(X, index=[i for i in range(X.shape[0])], columns=['f' + str(i) for i in range(X.shape[1])])
    #print(pd.isnull(dfX).sum())
    # concat tweet vector and account vector....................................
    result_X = pd.concat([dfX, X_account], axis=1, join='inner')
    #print(pd.isnull(result_X).sum())

    result_X.to_csv('data\\AllFeature_special.csv', index=None, header=True, sep=",")

    clf = joblib.load(classificationModel)
    y_pred = clf.predict(result_X)
    with open(outputPredictionFile, 'w', encoding="utf-8") as filehandle:
        filehandle.writelines("%s\n" % place for place in y_pred)
    return y_pred


def applyAccountTweetInfoModel(Testfile_profile,Testfile_tweet, classificationModel, outputPredictionFile):

    # X_account is account feature vector....................................

    X_account = pd.read_csv(Testfile_profile, sep=",", header=None)


    #print(pd.isnull(dfX).sum())
    # concat tweet vector and account vector....................................
    result_X = X_account
    #print(pd.isnull(result_X).sum())


    clf = joblib.load(classificationModel)
    y_pred = clf.predict(result_X)
    with open(outputPredictionFile, 'w', encoding="utf-8") as filehandle:
        filehandle.writelines("%s\n" % place for place in y_pred)
    return y_pred


def applyTweetFeatureModel_special(Testfile_profile,Testfile_tweet, classificationModel,botWord,humanWord, outputPredictionFile):
    botWord = joblib.load(botWord)
    humanWord = joblib.load(humanWord)
    tf,special, hashtag,embed = preprocessingandSpecial(botWord,humanWord,Testfile_tweet)
    X=special
    # X_account is account feature vector....................................

    df_account = pd.read_csv(Testfile_profile, sep=",", header=None)
    df_account = df_account.drop(columns=[10,11,12,13,14,15,16,17,18,19,20])

    dfX = pd.DataFrame(X, index=[i for i in range(X.shape[0])], columns=['f' + str(i) for i in range(X.shape[1])])
    #print(pd.isnull(dfX).sum())
    # concat tweet vector and account vector....................................
    result_X = pd.concat([dfX, df_account], axis=1, join='inner')
    #print(pd.isnull(result_X).sum())

    result_X.to_csv('data\\TweetFeature_special.csv', index=None, header=True, sep=",")

    clf = joblib.load(classificationModel)

    y_pred = clf.predict(result_X)
    with open(outputPredictionFile, 'w', encoding="utf-8") as filehandle:
        filehandle.writelines("%s\n" % place for place in y_pred)
    return y_pred

def applyTweetInfoModel(Testfile_profile, classificationModel, outputPredictionFile):
    # X_account is account feature vector....................................

    df_account = pd.read_csv(Testfile_profile, sep=",", header=None)
    df_account = df_account.drop(columns=[10,11,12,13,14,15,16,17,18,19,20])

    df_account.to_csv('data\\TweetInfoFeature.csv', index=None, header=True, sep=",")

    clf = joblib.load(classificationModel)
    y_pred = clf.predict(df_account)
    with open(outputPredictionFile, 'w', encoding="utf-8") as filehandle:
        filehandle.writelines("%s\n" % place for place in y_pred)
    return y_pred
