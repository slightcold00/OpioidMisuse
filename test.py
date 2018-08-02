import numpy as np
from predeal import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

MAX_SEQUENCE_LENGTH= 5
all_labels = [0,1]

text1='some thing to eat'
text2='some thing to drink'
all_texts=[text1,text2]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print(type(all_labels))
print('Shape of label tensor:', labels.shape)

print(np.shape(trn_data[1]))
print(word_index)