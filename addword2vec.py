#coding:utf-8
from gensim.models import Word2Vec
import logging,gensim,os
from gensim.models.word2vec import LineSentence

# 加载模型
model = Word2Vec.load('web_words.model')

# 导入新增语料
new_corpus = LineSentence('data/percocet_word.txt')

# 在线训练模型
model.train(new_corpus,total_examples=model.corpus_count,epochs=model.epochs)

# 保存新模型
model.save('data/web_words.model')
model.wv.save_word2vec_format('data/web_words.vector', binary=False)