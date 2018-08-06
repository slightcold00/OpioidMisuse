import tensorflow as tf

import mytokenize
import csv
import numpy as np

from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

wordvec = open('web_words.vector','r+')


word_index = {}
r = 0
for row in wordvec:
    te = row.split(' ')
    key = te[0]
    word_index[key] = r+1
    te.remove(te[0])
    results = list(map(float,te))
    r += 1 

trn_seq = list()
lable_seq = list()
test_seq = list()
lable_test_seq = list()

#read raw data
f = open('data/lableset.csv')
f_csv = csv.DictReader(f)
t = open('data/test.csv')
t_csv = csv.DictReader(t)

def word_to_index(document):
    sequence = list()
    index = 0
    for word in document:
        try:
            index = word_index[word]
            if not(index == 0):
                sequence.append(index)
        except:
            continue
    return sequence

for row in f_csv:
    raw_text = row['text']
    token_text = mytokenize.tokenize(raw_text)
    tweet_text = word_to_index(token_text)
    trn_seq.append(tweet_text)
    lable_seq.append(row['abuse'])

for row in t_csv:
    raw_test = row['text']
    token_test = mytokenize.tokenize(raw_test)
    tweet_test = word_to_index(token_test)
    test_seq.append(tweet_test)
    lable_test_seq.append(row['abuse'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(trn_seq)
tokenizer.word_index = word_index

trn_data = tokenizer.sequences_to_matrix(trn_seq, mode='tfidf')
lable_data = to_categorical(np.asarray(lable_seq))
test_data = tokenizer.sequences_to_matrix(test_seq, mode='tfidf')
lable_test = to_categorical(np.asarray(lable_test_seq))

model = Sequential()
model.add(Dense(512, input_shape=(len(word_index)+1,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(lable_data.shape[1], activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop')
model.fit(trn_data, lable_data,validation_data=(trn_data, lable_data), epochs=3, batch_size=100)
model.save('word_vector_mlp.h5')

preds = model.predict(test_data)
decoded = tf.argmax(preds, axis=1)
with tf.Session() as sess:
    preds = sess.run(decoded)
print('preds',preds)

for j in range(len(lable_test_seq)):
    lable_test_seq[j] = int(lable_test_seq[j])
print('lable',lable_test_seq)

TP = 0
FN = 0
TN = 0
FP = 0

for i in range(len(preds)):
    if int(lable_test_seq[i]) == 1:
        if preds[i] == 1:
            TP += 1
        else:
            FN += 1
    else:
        if preds[i] == 1:
            FP += 1
        else:
            TN += 1

print(int(TP+TN)/int(len(lable_test_seq)))        
precision = int(TP)/int(TP+FP)
recall = int(TP)/int(TP+FN)
F_1 = int(2*TP)/int(2*TP + FP + FN)

print("precision",precision)
print('recall',recall)
print('F_1',F_1)
