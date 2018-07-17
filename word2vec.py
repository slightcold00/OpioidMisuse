import logging
import os.path
import sys
import multiprocessing
import csv

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import tokenize

output = open('data/words.txt', 'w+')
#read raw data
f = open('data/codeine_noRT.csv')
f_csv = csv.DictReader(f)

space = ' '

for row in f_csv:
    raw_text = row['text']
    token_text = tokenize.tokenize(raw_text)
    output.write(space.join(token_text) + '\n')
