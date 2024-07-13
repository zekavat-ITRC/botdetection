#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import requests
import xlsxwriter
import warnings
from sys import argv
import pandas
import Preprocessing
import applyModels
import pandas

from writeToMySQL import connectTodb, insertToDB, searchDB, emptyDB


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


def read_tweets(startDate , endDate, *args):
    print(startDate)
    print(endDate)
    query = "SELECT id,publisher,copies,comments,likes,content_text,shdate,publish_id,mentions FROM socialmedia_v5 WHERE source='twitter-tweets'  AND shdate>= '{}' AND shdate <='{}' AND likes>=500".format(startDate , endDate)
    publisher,all = get_result_tweet({"query": query})
    print("we are getting tweets in during predefined date. number of tweet is: ", len(all))
    print("we are getting tweets in during predefined date. number of distinct users is: ", len(publisher))
    return publisher,all


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


def get_result_tweet(param):
        sess,url,headers = connectToserver()
        req = sess.post(url, data=json.dumps(param), headers=headers,verify=False)
        result = req.json()
        curs = ""
        more = True
        j = 0

        results = []
        headert=["id","publisher","copies","comments","likes","content_text","shdate","publish_id","mentions"]
        results.append(headert)
        publisher_List=[]
        while more:

            for i in range(len(result['rows'])):
              if(str(result['rows'][i][0])==("tw"+str(result['rows'][i][7]))):

                results.append(result['rows'][i])

                publisher_List.append(result['rows'][i][1])
                j+=1
                if j%100000==0:
                    print("\t" + str(int(j/100000)) + "00 k")
            try:
                curs = result['cursor']
                param = {"cursor": result['cursor']}
                req = sess.post(url,data=json.dumps(param), headers=headers, verify=False)
                result = req.json()
            except Exception:
                more = False
        sess.close()
        uniquePublisher = unique(publisher_List)
        return uniquePublisher,results


def get_result_profile(sess,url,headers,param):

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


def Sort(sub_li):
    return (sorted(sub_li, key=lambda x: x[1]))




publisherResults, allTweetResults =read_tweets(*argv[1:])

with open('Testdata\\listfile_publisher.txt', 'w', encoding="utf-8") as filehandle:
    filehandle.writelines("%s\n" % place for place in publisherResults)

write_to_excel(allTweetResults,'Testdata\\listfile_all.xlsx')



j = 0
k=0
with open('Testdata\\listfile_publisher.txt', 'r') as rfilehandle:
    session, url, header=connectToserver()
    basicList = rfilehandle.readlines()
    headerp=["id", "username","follower","following","verified","no_posts","no_likes","join_date"]
    profiles=[]
    notprofile=['1']
    profiles.append(headerp)
    for line in basicList:
        currentPlace = line[0:-1]

        userInfo=read_profile(currentPlace,session, url, header)

        if userInfo:
            profiles.append(userInfo)
            j = j + 1
            if j % 1000 == 0:
                print("\t user found" + str(int(j / 1000)) + " k")
        else:
            notprofile.append(currentPlace)
            k= k+1
            if k % 1000 == 0:
                print("\t user notfound" + str(int(k / 1000)) + " k")
prof= Sort(profiles[1:])
with open("Testdata\\list_notprofiles.csv" ,mode='w', encoding='utf-8') as filenotprofile:
    filenotprofile.writelines("%s\n" % np for np in notprofile )

write_to_excel(profiles,'Testdata\\list_profiles.xlsx')
print("number of publisher that found in profile table :" + str(j))
print("number of publisher that not found in profile table :" + str(k))

folder="Testdata"
Preprocessing.preprocessingTestData("Testdata",'list_profiles.xlsx','listfile_all.xlsx','list_notprofiles.csv')

if argv[3]=="account":
    y_prediction = applyModels.applyAccountModel('Testdata\\AllTweetUser.csv','model\\account_classification_model.pkl','Testdata\\prediction_Account.txt')

elif argv[3]=="tweet":
    y_prediction = applyModels.applyTweetModel('Testdata\\AllTweetsPerUser2.csv','model\\tweet_classification_model.pkl','Testdata\\prediction_Tweet.txt','model\\botword.pkl','model\\humanword.pkl')


elif argv[3]=="tweetinfo":
    y_prediction = applyModels.applyTweetInfoModel('Testdata\\AllTweetUser.csv',
                                                    'model\\tweetinfo_classification_model.pkl',
                                                    'Testdata\\prediction_TweetInfo.txt')
elif argv[3]=="accounttweetinfo":
    y_prediction = applyModels.applyTweetInfoModel('Testdata\\AllTweetUser.csv',
                                                    'model\\AccountTweetInfo_classification_model.pkl',
                                                    'Testdata\\prediction_AccountTweetInfo.txt')
elif argv[3]=="alltweet":
    y_prediction = applyModels.applyTweetFeatureModel_special('Testdata\\AllTweetUser.csv', 'Testdata\\AllTweetsPerUser2.csv',
                                                    'model\\allTweetfeature_classification_model.pkl','model\\botword.pkl','model\\humanword.pkl',
                                                    'Testdata\\prediction_TweetFeatures_special.txt')

else:
    y_prediction = applyModels.applyAllFeatureModel_special('Testdata\\AllTweetUser.csv',
                                                            'Testdata\\AllTweetsPerUser2.csv',
                                                            'model\\allfeature_classification_model.pkl',
                                                            'model\\botword.pkl', 'model\\humanword.pkl',
                                                            'Testdata\\prediction_AllFeatures_special.txt')
user_counter=0
mydb, mycursor =connectTodb()
#emptyDB(mydb,mycursor)

for p in prof:
    userId=p[0]
    username=p[1]
    subject=argv[4]
    classificationmodel=argv[4]
    isbot=y_prediction[user_counter]
    print(username,userId)
    insertToDB(mydb,mycursor,userId,username,argv[1],argv[2],classificationmodel,subject,str(isbot))
    user_counter+=1




