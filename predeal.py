import tokenize
import csv
import numpy as np


trn_data = []

#read raw data
f = open('data/codeine_noRT.csv')
f_csv = csv.DictReader(f)


#将分词结果转化为向量
def doc2vec(document):
    # 100维的向量
    doc_vec = np.zeros(100)
    tot_words = 0

    for word in document:
        try:
        # 查找该词在预训练的word2vec模型中的特征值
            vec = np.array(lookup_bd.value.get(word)) + 1
            # print(vec)
            # 若该特征词在预先训练好的模型中，则添加到向量中
            if vec != None:
                doc_vec += vec
                tot_words += 1
        except:
            continue

    vec = doc_vec / float(tot_words)
    return vec



for row in f_csv:
    raw_text = row['text']
    token_text = tokenize.tokenize(raw_text)
    print(token_text)
