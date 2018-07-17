import tokenize
import csv
import numpy as np

#coding:utf-8
import gensim

trn_data = []

#read raw data
f = open('data/codeine_noRT.csv')
f_csv = csv.DictReader(f)

# 加载模型
model = gensim.models.Word2Vec.load('web_words.model')

#将分词结果转化为向量
def doc2vec(document):
    # 100维的向量
    vec = np.zeros(100)

    for word in document:
        try:
            vec = model[word]
        except:
            continue

    #vec = doc_vec / float(tot_words)
    return vec

for row in f_csv:
    raw_text = row['text']
    token_text = tokenize.tokenize(raw_text)
    tweet_text = doc2vec(token_text)
    trn_data.append(tweet_text)
