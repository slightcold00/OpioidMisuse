import numpy as np
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import rmsprop

import h5py
import tensorflow as tf

import mytokenize
import csv

TP = 0
FP = 0
FN = 0
TN = 0

MAX_SEQUENCE_LENGTH = 50 # 每条新闻最大长度
EMBEDDING_DIM = 100 # 词向量空间维度
WORDS = 4395
wordvec = open('web_words.vector','r+')

embedding_matrix = np.zeros((WORDS+1, EMBEDDING_DIM))

word_index = {}
r = 0
for row in wordvec:
    te = row.split(' ')
    key = te[0]
    word_index[key] = r+1
    te.remove(te[0])
    results = list(map(float,te))
    embedding_matrix[r] = np.asarray(results)
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

trn_data = pad_sequences(trn_seq, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_seq, maxlen=MAX_SEQUENCE_LENGTH)
lable_data = to_categorical(np.asarray(lable_seq))
lable_test = to_categorical(np.asarray(lable_test_seq))

embedding_layer = Embedding(WORDS+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
   

#buile te model
rmsprop=rmsprop(lr=1e-4)
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.1))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.summary()
#plot_model(model, to_file='model.png',show_shapes=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
model.fit(trn_data, lable_data, validation_data=(trn_data, lable_data), epochs=3, batch_size=100)
model.save('word_vector_cnn.h5')
print(model.evaluate(test_data, lable_test))

preds = model.predict(test_data)
decoded = tf.argmax(preds, axis=1)
with tf.Session() as sess:
    preds = sess.run(decoded)
print('preds',preds)

for j in range(len(lable_test_seq)):
    lable_test_seq[j] = int(lable_test_seq[j])
print('lable',lable_test_seq)



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