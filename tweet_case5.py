#!/usr/bin/python
# -*- coding: utf-8 -*-

import json

import mysql.connector
import requests
import xlsxwriter
import warnings
from sys import argv
import pandas
import Preprocessing
import applyModels
import pandas

#from writeToMySQL import connectTodb,insertToDB,searchDB
from writeToMySQL import insertToDB, emptyDB, emptyDBcase5, searchDB


def connectTodb():
  mydb = mysql.connector.connect(
    host="172.20.81.140",
    user="asadi",
    password="as@098",
    database="twitter"
  )
  print(mydb)
  mycursor = mydb.cursor()
  #mycursor.execute("SHOW TABLES")
  #print([column[0] for column in mycursor.fetchall()])

  mycursor.execute("SHOW columns FROM tweets_case5")

  print([column[0] for column in mycursor.fetchall()])

  return mydb, mycursor


def connectToserver():
    url = 'https://172.20.81.139:9200/_sql?format=json'
    headers = {"Content-Type": "application/json"}
    sess = requests.Session()
    sess.auth = ('asadi', 'as@098')
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    return sess,url,headers

def write_to_excel(result,filename):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    for row_num, row_data in enumerate(result):
        for col_num, col_data in enumerate(row_data):
            worksheet.write(row_num, col_num, col_data)

    workbook.close()


def read_tweets(mycursor):

    query = "SELECT id,user,retweets,replies,likes,txt,timestamp,conversation_id,mentions FROM tweets_case5"

    mycursor.execute(query)
    print("execute search........................................")
    myresult = mycursor.fetchall()
    print("fetch.................................................")
    publisher=[]
    all=[]
    headert=["id","publisher","copies","comments","likes","content_text","shdate","publish_id","mentions"]
    all.append(headert)
    for x in myresult:
        if x[0]==x[7]:
            publisher.append(x[1])
            all.append(x)
  #  publisher,all = get_result_tweet({"query": query})
    uniqpublisher = unique(publisher)
    mydb.disconnect()
    print("we are getting tweets in during predefined date. number of tweet is: ", len(all))
    print("we are getting tweets in during predefined date. number of distinct users is: ", len(uniqpublisher))
    return uniqpublisher,all


def read_profile(publisher,session,url,headers):

    query = "SELECT id, username,follower,following,verified,no_posts,no_likes,join_date FROM profiles  WHERE  profiles.source='twitter' AND username='{}'".format(publisher)
    posts = get_result_profile(session,url,headers,{"query": query})
    #avalin khoroji [0] jadid tarin moshakhasat bedast amade az user ast
    if (len(posts)>0):
        return posts[0]


def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list



def get_result_profile(sess,url,headers,param):
    sess, url, headers = connectToserver()

    req = sess.post(url, data=json.dumps(param), headers=headers, verify=False)
    result = req.json()
    curs = ""
    more = True
    j = 0

    results = []
    while more:
        for i in range(len(result['rows'])):
                results.append(result['rows'][i])
                j += 1
                if j % 100000 == 0:
                    print("\t" + str(int(j / 100000)) + "00 k")
        try:
            curs = result['cursor']
            param = {"cursor": result['cursor']}
            req = sess.post(url, data=json.dumps(param), headers=headers, verify=False)
            result = req.json()
        except Exception:
            more = False
    sess.close()
    return results



mydb,mycursor= connectTodb()
publisherResults, allTweetResults=read_tweets(mycursor)



folder="casedata"
with open(folder+"\\"+'listfile_publisher.txt', 'w', encoding="utf-8") as filehandle:
    filehandle.writelines("%s\n" % place for place in publisherResults)

write_to_excel(allTweetResults,'casedata\\listfile_all.xlsx')


def Sort(sub_li):

    # key is set to sort using second element of

    return (sorted(sub_li, key=lambda x: x[1]))


j = 0
with open('casedata\\listfile_publisher.txt', 'r') as rfilehandle:
    session, url, header=connectToserver()
    basicList = rfilehandle.readlines()
    headerp=["id", "username","follower","following","verified","no_posts","no_likes","join_date"]
    profiles=[]
    profiles.append(headerp)
    notFounduser=[]
    notFounduser.append(1)
    for line in basicList:
        currentPlace = line[0:-1]

        userInfo=read_profile(currentPlace,session, url, header)

        if userInfo:
            profiles.append(userInfo)
            j = j + 1
            if j % 1000 == 0:
                print("\t" + str(int(j / 1000)) + " k")
        else:
            notFounduser.append(currentPlace)
prof= Sort(profiles[1:])
write_to_excel(profiles,'casedata\\list_profiles.xlsx')
with open(folder+"\\"+'list_notprofiles.csv', 'w', encoding="utf-8") as filehandle2:
    filehandle2.writelines("%s\n" % place for place in notFounduser)

#write_to_excel(notFounduser,'casedata\\list_notprofiles.csv')

print("number of publisher that find in profile table :" + str(j))

print("start preprocessing...........................")
folder="casedata"
Preprocessing.preprocessingcasedata(folder,'list_profiles.xlsx','listfile_all.xlsx','list_notprofiles.csv')

if argv[1]=="account":
    y_prediction = applyModels.applyAccountModel('casedata\\AllTweetUser.csv','model\\account_classification_model.pkl','casedata\\prediction_Account.txt')

elif argv[1]=="tweet":
    y_prediction = applyModels.applyTweetModel('casedata\\AllTweetsPerUser2.csv','model\\tweet_classification_model.pkl','casedata\\prediction_Tweet.txt','model\\botword.pkl','model\\humanword.pkl')

elif argv[1]=="tweetinfo":
    y_prediction = applyModels.applyTweetInfoModel('casedata\\AllTweetUser.csv',
                                                    'model\\tweetinfo_classification_model.pkl',
                                                    'casedata\\prediction_TweetInfo.txt')
elif argv[1]=="accounytweetinfo":
    y_prediction = applyModels.applyTweetInfoModel('casedata\\AllTweetUser.csv',
                                                    'model\\AccountTweetInfo_classification_model.pkl',
                                                    'casedata\\prediction_AccountTweetInfo.txt')
elif argv[1]=="alltweet":
    y_prediction = applyModels.applyTweetFeatureModel_special('casedata\\AllTweetUser.csv', 'casedata\\AllTweetsPerUser2.csv',
                                                    'model\\allTweetfeature_classification_model.pkl','model\\botword.pkl','model\\humanword.pkl',
                                                    'casedata\\prediction_TweetFeatures_special.txt')

else:
    y_prediction = applyModels.applyAllFeatureModel_special('casedata\\AllTweetUser.csv',
                                                            'casedata\\AllTweetsPerUser2.csv',
                                                            'model\\allfeature_classification_model.pkl',
                                                            'model\\botword.pkl', 'model\\humanword.pkl',
                                                            'casedata\\prediction_AllFeatures_special.txt')

user_counter=0
mydb, mycursor =connectTodb()
#emptyDB(mydb,mycursor)
searchDB(mycursor)

for p in prof:
    userId=p[0]
    username=p[1]
    isbot=y_prediction[user_counter]
    insertToDB(mydb,mycursor,userId,username,'1399-04-25','1399-05-03',argv[1],"edam_nakonid",str(isbot))
    user_counter+=1




