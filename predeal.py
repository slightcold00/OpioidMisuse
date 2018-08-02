# -*- coding: utf-8 -*-
import mytokenize
import csv
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark import SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint

import gensim
import pandas as pd

trn_data = []
lable_data = []
test_data=[]
lable_test = []

#read raw data
f = open('data/lableset.csv')
f_csv = csv.DictReader(f)
t = open('data/test.csv')
t_csv = csv.DictReader(t)

# 加载模型
model = gensim.models.Word2Vec.load('web_words.model')

#将分词结果转化为向量
def doc2vec(document):
    # 100维的向量
    dec = np.zeros(100)
    word_vec = np.zeros(100)
    num = 0
    a = np.zeros(100)

    for word in document:
        try:
            word_vec = model[word]
            if not((word_vec == a).all()):
                dec += word_vec
                num += 1
        except:
            continue
    vec = dec/num      
    return vec
    

for row in f_csv:
    raw_text = row['text']
    token_text = mytokenize.tokenize(raw_text)
    tweet_text = doc2vec(token_text)
    trn_data.append(tweet_text)
    lable_data.append(row['abuse'])

for row in t_csv:
    raw_test = row['text']
    token_test = mytokenize.tokenize(raw_test)
    tweet_test = doc2vec(token_test)
    test_data.append(tweet_test)
    lable_test.append(row['abuse'])

k=0
for j in lable_data:
    if int(j) == 1:
        k += 1 
        #print("k",k)
print(k) 
print(k/len(lable_data))