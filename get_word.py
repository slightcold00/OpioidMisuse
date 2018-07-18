import csv
import mytokenize

output = open('data/codeine_word2.txt', 'w+')
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