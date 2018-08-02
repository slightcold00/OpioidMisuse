from gensim.models import word2vec
import os
import gensim

model_1 = word2vec.Word2Vec.load('web_words.model')
# 计算两个词的相似度/相关程度
y1 = model_1.similarity("codeine", "fentanyl")
print(u"the similarity of codeine and fentanyl：", y1)
print("-------------------------------\n")

# 计算某个词的相关词列表
y2 = model_1.most_similar("codeine", topn=50)  # 10个最相关的
print(u"the most related word with codeine：\n")
for item in y2:
    print(item[0], item[1])
print("-------------------------------\n")