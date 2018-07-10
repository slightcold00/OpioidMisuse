#导入tweepy
import tweepy
import json

#抓取次数
MAX_QUERIES = 2
 
#提交你的Key和secret
auth = tweepy.OAuthHandler(setting.consumer_key, setting.consumer_secret)
auth.set_access_token(setting.access_token, setting.access_token_secret)
 
#获取类似于内容句柄的东西
api = tweepy.API(auth)

output_file = open('codeine.txt','w+')

i = MAX_QUERIES
tweet_id = []
MAX_ID = 10

while i > 0:
    if MAX_ID == 10:
        #按关键字搜索(q = 关键字 ,count = 返回的数据量 . 推特一次最多返回100条)
        search_results = api.search(q='codeine',lang='en',count=100)
        
        #对对象进行迭代
        for tweet in search_results:
            tweet_id.append(tweet._json["id"])
            #tweet还是一个对象,推特的相关信息在tweer._json里
            #这里是检测消息是否含有'text'键,并不是所有TWitter返回的所有对象都是消息(有些可能是用来删除消息或者其他内容的动作--这个没有确认),区别就是消息对象中是否含有'text'键
            if 'text' in tweet._json:
                print(tweet._json['user']['screen_name'],tweet._json['text'])
                #这里是把用户和内容打印出来,如果需要保存到文件需要用json库的dumps函数转换为字符串形式后写入到文件中
                output_file.write(json.dumps('@'+tweet._json['user']['screen_name'])+': '+json.dumps(tweet._json['text'])+'\n')
        MAX_ID = min(tweet_id)
    
    else:
        search_results = api.search(q='codeine', count=100, lang='en', max_id = MAX_ID-1)
        for tweet in search_results:
            tweet_id.append(tweet._json["id"])
            if 'text' in tweet._json:
                print(tweet._json['user']['screen_name'],tweet._json['text'])
                output_file.write('@'+json.dumps(tweet._json['user']['screen_name'])+': '+json.dumps(tweet._json['text'])+'\n')
        MAX_ID = min(tweet_id)
    
    i -= 1

output_file.close()