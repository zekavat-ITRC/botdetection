
import datetime
from collections import Counter

import pandas as pd

def convertTimeStampToDays(row):
    timestamp = row['join_date']
    readable=datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").date()
    today=datetime.date.today()
    Age = today-readable
    #print(Age.days)
    return Age.days


def idEidConv (row):
   if str(row[0]) == ('tw'+str(row[8])) :
      return 1
   else:
      return 0

def tostr (row):
    return str(row[6])

def findlen (row):
   if not pd.isnull(row[6]):
     if row[6]!=0:
      return len(row[6])
     else:
         return 0
   else:
      return 0

def mention (row):
   if not pd.isnull(row[9]) :
      return 1
   else:
      return 0


def preprocessingTestData (folder,prifile_fname,tweet_fname,notfound):

    df_profile = pd.read_excel(folder+"\\"+prifile_fname,names=["id",1,"follow","following","verified","tweets","likes","join_date"])

    df_profile['Age']= df_profile.apply(lambda row: convertTimeStampToDays(row), axis=1)
    df_profile['follow/following']= df_profile['follow']/df_profile['following']
    df_profile['follow/Age']=df_profile['follow']/df_profile['Age']
    df_profile['following/Age']=df_profile['following']/df_profile['Age']
    df_profile['tweet/Age']=df_profile['tweets']/df_profile['Age']
    df_profile['like/Age']=df_profile['likes']/df_profile['Age']
    df_profile=df_profile.drop(columns='join_date')
    df_profile["follow/following"] = df_profile["follow/following"].fillna(1)
    df_profile["verified"] = df_profile["verified"].fillna(0)
    df_profile.to_excel(folder+'\\profiles.xlsx')

    df_tweet= pd.read_excel(folder+"\\"+tweet_fname,names=[0,1,3,4,5,6,7,8,9])
#    df_tweet[6]=df_tweet[6]
    df_tweet['idEidConv'] = df_tweet.apply(lambda row: idEidConv(row), axis=1)
    df_tweet['length'] = df_tweet.apply(lambda row: findlen(row), axis=1)
    df_tweet['hasmention'] = df_tweet.apply(lambda row: mention(row), axis=1)

    df_tweet[6] = df_tweet.apply(lambda row: tostr(row), axis=1)
    df_tweet = df_tweet.rename(columns={1: "1", 3: "3", 4: "4", 5: "5",6:"6", 9: "9"})

    # grouping the tweets of a user
    grp = df_tweet.fillna('').groupby("1")
    # tweets are join with &

    df_agg = grp.agg({"3": 'sum', "4": 'sum', "5": 'sum', "1": 'count',"hasmention": 'sum',  'idEidConv': 'sum', 'length': ['mean', 'std'],"6": ' & '.join})
    # df_agg= grp.agg({"3" : 'sum', "4" : 'sum', "5": 'sum', "9": ' & '.join,  'idEidConv': 'sum','length':['mean','std']})
    df_agg.columns = ["_".join(x) for x in df_agg.columns.ravel()]

    df_agg.head()

    df_agg["length_std"] = df_agg["length_std"].fillna(1)

    tweet_agg = grp.agg({"6": ' & '.join})
    # check null
    #print(pd.isnull(tweet_agg).sum())

    tweet_word = tweet_agg["6"].str.split()

    # Get amount of unique words
    tweet_agg['tweet_uniqword'] = tweet_word.apply(set).apply(len)

    # Get amount of words
    tweet_agg['tweet_word'] = tweet_word.apply(len)

    results = set()
    tweet_agg["6"].str.split().apply(results.update)

    results = Counter()
    count_unique = tweet_word.apply(results.update)
    unique_counts = sum((results).values())
    tweet_agg['uniquewordcount'] = unique_counts

    df_agg['richness'] = (tweet_agg['tweet_uniqword'] * 1000) / tweet_agg['uniquewordcount']
    df_agg['distinct_word'] = tweet_agg['tweet_uniqword'] / tweet_agg['tweet_word']
    tweet_agg = tweet_agg.drop(columns="tweet_uniqword")
    tweet_agg = tweet_agg.drop(columns="tweet_word")
    tweet_agg = tweet_agg.drop(columns="uniquewordcount")


    result=df_agg.join(df_profile.set_index(1),how='inner')
    tweet_df=result['6_join']

    result=result.drop(columns="6_join")
    result=result.drop(columns="id")
    result.to_excel(folder+'\\AllTweetUser.xlsx')
    result.to_csv (folder+'\\AllTweetUser.csv', index = None, header=None, sep=",")
    tweet_df = tweet_df.to_frame().reset_index()
   # tweet_df=tweet_df.rename(columns=)
   # tweet_df.index.name = 1

    notfound=pd.read_csv(folder+"\\"+notfound,encoding='utf-8')
  #  notfound=notfound.rename(columns={1:"index"})
    resNotfound = pd.merge(notfound,tweet_agg,left_on='1',right_index=True, how = 'inner')
#    resNotfound=tweet_df.join(notfound ,,how='inner')
    nottweet_df = resNotfound['6']
    nottweet_df.to_excel(folder + "\\" + 'AllTweetsPerUserNot.xlsx', encoding='utf-8')
    nottweet_df.to_csv(folder + "\\" + 'AllTweetsPerUserNot.csv', encoding='utf-8', header=True, sep=",")

#    df_tweet.to_excel("data\\AllTweetTest.xlsx")
    tweet_df.to_excel(folder+"\\"+'AllTweetsPerUser2.xlsx', encoding='utf-8')
    tweet_df.to_csv(folder+"\\"+'AllTweetsPerUser2.csv', encoding='utf-8',header=True, sep=",")

    df_agg.to_excel(folder+"\\"+'AllTweetsPerUser.xlsx', encoding='utf-8')
    df_agg.to_csv(folder+'\\AllTweetsPerUser.csv', encoding='utf-8',header=True, sep=",")



