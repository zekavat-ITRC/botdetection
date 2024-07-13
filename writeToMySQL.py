from datetime import datetime

import mysql.connector

def connectTodb():
  mydb = mysql.connector.connect(
    host="172.20.81.140",
    user="asadi",
    password="as@098",
    database="twitter"
  )
  print(mydb)
  mycursor = mydb.cursor()
  mycursor.execute("SHOW TABLES")

  print([column[0] for column in mycursor.fetchall()])
  mycursor.execute("SHOW columns FROM tweets_case5")
  print([column[0] for column in mycursor.fetchall()])
  mycursor.execute("SHOW columns FROM bot_users")
  print([column[0] for column in mycursor.fetchall()])
  return mydb, mycursor


def insertToDB(mydb,mycursor,id,username,startDate,endDate,classificationmodel, subject,isbot):
  date_time_start = datetime.strptime(startDate ,'%Y-%m-%d')
  date_time_end = datetime.strptime(endDate, '%Y-%m-%d')

  if date_time_end >= date_time_start :
    sql = "INSERT INTO bot_users (userid,user,startdate,enddate,detectdate,classification_model,subject,isbot) VALUES (%s, %s,%s,%s,%s,%s,%s,%s)"
    now = datetime.now()
    val = (id[2:], username, date_time_start, date_time_end, now,classificationmodel,subject, isbot)
    mycursor.execute(sql, val)
    mydb.commit()
    #print(mycursor.rowcount, "record inserted.")
  else:
    print("endDate is less that startDate")


def searchDBCount(mycursor):
  mycursor.execute("SELECT COUNT(*) FROM bot_users WHERE isbot='1'")
  res =mycursor.fetchone()
  print(res[0])
  #myresult = mycursor.rowcount()

def searchDB(mycursor):
  mycursor.execute("SELECT * FROM bot_users")
  myresult = mycursor.fetchall()
  print(len(myresult))
  for x in myresult:
    print(x)


def emptyDB(mydb,mycursor):
  Delete_all_rows = """truncate table bot_users """
  mycursor.execute(Delete_all_rows)
  mydb.commit()

def emptyDBcase5(mydb,mycursor):
  Delete_all_rows = """truncate table tweets_case5"""
  mycursor.execute(Delete_all_rows)
  mydb.commit()

def deleteSomeRowDB(mydb,mycursor, startDate,endDate):
  date_time_start = datetime.strptime(startDate, '%Y-%m-%d')
  date_time_end = datetime.strptime(endDate, '%Y-%m-%d')

  if date_time_end>=date_time_start:

    sql = "DELETE FROM bot_users WHERE startdate=%s and enddate=%s"
    val = (date_time_start, date_time_end)
    mycursor.execute(sql,val)
    mydb.commit()
  else:
    print("endDate is less that startDate")

def deleteSomeRowDetectdate(mydb,mycursor, detectdate):
  date_time = datetime.strptime(detectdate, '%Y-%m-%d')
  sql = "DELETE FROM bot_users WHERE detectdate=%s"
  val = (date_time)
  mycursor.execute(sql,val)
  mydb.commit()

'''
mydb,mycursor=connectTodb()

#searchDBcase5(mycursor)
#searchDBCount(mycursor)

emptyDB(mydb,mycursor)
insertToDB(mydb,mycursor,"02111111111","dskkkd",'1398-12-14','1398-12-14','all','corona','1')
insertToDB(mydb,mycursor,"1121","dsd",'1398-12-14','1398-12-14','1')
insertToDB(mydb,mycursor,"11211","dsd",'1398-12-24','1398-12-14','1')
insertToDB(mydb,mycursor,"11211111","dsd",'1398-12-24','1398-12-24','1')
searchDB(mycursor)
#deleteSomeRowDB(mydb,mycursor,'1398-12-14','1398-12-14')
#searchDB(mycursor)

'''