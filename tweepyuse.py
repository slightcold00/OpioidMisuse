#导入tweepy
import tweepy
import json
from private import *
import pymysql
import datetime

#抓取次数
MAX_QUERIES = 400
 
#提交你的Key和secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
#获取类似于内容句柄的东西
api = tweepy.API(auth)

i = MAX_QUERIES
tweet_id = []
MAX_ID = 10


#连接数据库
def executeSql(sql,values):
    conn = pymysql.connect(host = host,port = port,user = user,passwd = passwd,db = db)
    cursor = conn.cursor()
    conn.set_charset('utf8')
    effect_row = cursor.execute(sql, values)
    # 提交，不然无法保存新建或者修改的数据
    conn.commit()
    # 关闭游标
    cursor.close()
    # 关闭连接
    conn.close()

GMT_FORMAT = '%a %b %d %H:%M:%S %z %Y'
def insertNews(search_results):
    for tweet in search_results:
        data = tweet._json
        tweetObj = [json.dumps(data["id"]),datetime.datetime.strptime(data["created_at"],GMT_FORMAT),
                    json.dumps(data["user"]["screen_name"]),
                    json.dumps(data["favorite_count"]),json.dumps(data["retweet_count"]),json.dumps(data["text"]),json.dumps(data["source"]),
                    json.dumps(data["place"]["country_code"] if data['place'] != None else 'NULL'),json.dumps(data["user"]["location"]),
                    json.dumps(data["coordinates"]["coordinates"][0] if data["coordinates"] != None else 'NULL'),
                    json.dumps(data["coordinates"]["coordinates"][1] if data["coordinates"] != None else 'NULL')]
 
        sql = "insert into oxycontin(_id,created_at,screen_name,favorite_count,retweet_count,text,source,country_code,location,latitude,longitude)"\
             "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

        executeSql(sql=sql,values=tweetObj)
        
        tweet_id.append(tweet._json["id"])
        
    return tweet_id



while i > 0:
    if MAX_ID == 10:
        #按关键字搜索(q = 关键字 ,count = 返回的数据量 . 推特一次最多返回100条)
        search_results = api.search(q='oxycontin',lang='en',count=100)
        
        tweet_id = insertNews(search_results)

        MAX_ID = min(tweet_id)
    
    else:
        search_results = api.search(q='oxycontin', count=100, lang='en', max_id = MAX_ID-1,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        
        tweet_id = insertNews(search_results)

        MAX_ID = min(tweet_id)
    
    print(i)
    i -= 1