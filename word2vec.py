import logging
import os.path
import sys
import multiprocessing
import csv

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import mytokenize

output = open('data/words.txt', 'w+')
#read raw data
f = open('data/codeine_noRT.csv')
f_csv = csv.DictReader(f)

space = ' '

for row in f_csv:
    raw_text = row['text']
    token_text = mytokenize.tokenize(raw_text)
    output.write(space.join(token_text) + '\n')

f.close()
output.close()

inp = 'data/words.txt'
outp1 = 'web_words.model'
outp2 = 'web_words.vector'

# Word2Vec函数的参数：
# size 表示特征向量维度，默认100
# window 表示当前词与预测词在一个句子中的最大距离
# min_count 词频少于min_count次数的单词会被丢弃掉, 默认值为5
model = Word2Vec(LineSentence(inp), size=100, window=5, min_count=5,\
                     workers=multiprocessing.cpu_count())

# 默认格式model
model.save(outp1)
# 原始c版本model
model.wv.save_word2vec_format(outp2, binary=False)