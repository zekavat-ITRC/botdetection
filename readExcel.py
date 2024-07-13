from collections import Counter

import pandas as pd

#check tweet_id is equal conversation_id
def idEidConv (row):
   if row[0] == row[8] :
      return 1
   else:
      return 0

def mention (row):
   if not pd.isnull(row[9]) :
      return 1
   else:
      return 0

def findlen (row):
   if not pd.isnull(row[6]):
      return len(row[6])
   else:
      return 0
'''
#read tweet File .... large file
df1 = pd.read_excel('data\\tweetL.xlsx', sheet_name='info')
df1 = df1.drop(columns=['Unnamed: 0'])
df1 = df1.drop(columns=[12, 13, 14, 15])

#read tweet File .... small file
df2=pd.read_excel('data\\tweetS.xlsx',sheet_name='info')
df2=df2.drop(columns=['Unnamed: 0'])
df2=df2.drop(columns=[12])

# merge large and small File and remove duplicates...................

df3 = pd.concat([df1,df2]).drop_duplicates(subset=0).reset_index(drop=True)
df3=df3.drop(columns=[11])

# write all uniq  tweets ................
df3.to_excel('data\\AllTweet2.xlsx')

'''
df_tweet=pd.read_excel('data\\AllTweet2.xlsx',sheet_name='Sheet1')

df_tweet=df_tweet.drop(columns=2)
df_tweet=df_tweet.drop(columns=10)
df_tweet['idEidConv'] = df_tweet.apply(lambda row: idEidConv(row), axis=1)
df_tweet['length'] = df_tweet.apply(lambda row: findlen(row), axis=1)
df_tweet['hasmention'] = df_tweet.apply(lambda row: mention(row), axis=1)

df_tweet.to_excel('data\\AllTweet3.xlsx')


df_tweet=pd.read_excel('data\\AllTweet3.xlsx',sheet_name='Sheet1')
#read Account File

df_user=pd.read_excel('data\\users.xlsx',sheet_name='Sheet1')
df_user["follow/following"] = df_user["follow/following"].fillna(1)
df_user.to_excel('data\\users.xlsx')
df_tweet=df_tweet.rename(columns={1:"1", 3: "3", 4: "4",5:"5",9:"9",6:"6"})

#grouping the tweets of a user
grp=df_tweet.fillna('').groupby("1")
#tweets are join with &



df_agg= grp.agg({"3" : 'sum', "4" : 'sum', "5": 'sum', "hasmention": 'sum', "1" :'count', 'idEidConv': 'sum','length':['mean','std'],"6" : ' & '.join})
df_agg.columns = ["_".join(x) for x in df_agg.columns.ravel()]

df_agg.head()

df_agg["length_std"] = df_agg["length_std"].fillna(1)



#join with account file

#All tweets of a user (without other fields) ...............
tweet_agg= grp.agg({"6" : ' & '.join})
#check null
print(pd.isnull(tweet_agg).sum())


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

df_agg['richness']=(tweet_agg['tweet_uniqword']*1000)/tweet_agg['uniquewordcount']
df_agg['distinct_word']=tweet_agg['tweet_uniqword']/tweet_agg['tweet_word']
#tweet_agg['richness']=tweet_agg['tweet_uniqword']/tweet_agg['uniquewordcount']
#tweet_agg['distict_word']=tweet_agg['tweet_uniqword']/tweet_agg['tweet_word']

tweet_agg=tweet_agg.drop(columns="tweet_uniqword")
tweet_agg=tweet_agg.drop(columns="tweet_word")
tweet_agg=tweet_agg.drop(columns="uniquewordcount")


tweet_agg.to_excel('data\\AllTweetsPerUser.xlsx')
result=df_agg.join(df_user.set_index(1))
#result2=result.join(tweet_agg.set_index(1),how='inner')

result.to_excel('data\\AllTweetUser.xlsx')
'''
tweets = result[["3_sum","4_sum","5_sum","hasmention_sum","1_count","6_join","richness","distinct_word","length_mean","length_std","idEidConv_sum"]]
users = result[["follow","following","verified","tweets","likes","Age","follow/following","follow/age","following/age","likes/age","tweet/age"]]

users.to_csv('data\\userInfo.csv',encoding='utf-8',sep=",")
tweets.to_csv('data\\TweetInfo.csv',encoding='utf-8',sep=",")
'''

