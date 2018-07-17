from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator


text = open('data/codeine_word.txt','r').read()



wc = WordCloud(background_color="white", #背景颜色 
max_words=200,# 词云显示的最大词数
width=1000, height=860, margin=2,
stopwords=STOPWORDS.add('codeine'))
# 生成词云, 可以用generate输入全部文本(中文不好分词),也可以我们计算好词频后使用generate_from_frequencies函数
wc.generate(text)
# wc.generate_from_frequencies(txt_freq)
# txt_freq例子为[('词a', 100),('词b', 90),('词c', 80)]

# 以下代码显示图片
plt.imshow(wc)
plt.axis("off")
# 绘制词云
plt.figure()
plt.show()

# 保存图片
wc.to_file("picture/codeine_word.png")