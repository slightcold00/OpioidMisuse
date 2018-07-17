from gensim.models import Word2Vec
import logging,gensim,os
from gensim.models.word2vec import LineSentence

# 加载模型
model = Word2Vec.load('data/web_words.model')

# 导入新增语料
new_corpus = LineSentence('data/words2.txt')

# 在线训练模型
model.train(new_corpus)

# 保存新模型
model.save('data/web_words2.model')