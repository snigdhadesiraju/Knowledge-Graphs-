# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 07:30:50 2021

@author: snigh
"""

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')

scrapped_data=urllib.request.urlopen('https://en.wikipedia.org/wiki/Coronavirus_disease_2019')
article = scrapped_data .read()

parsed_article=bs.BeautifulSoup(article,'lxml')
paragraphs=parsed_article.find_all('p')
article_text=''

for p in paragraphs:
    article_text += p.text

article_text

# Cleaing the text
processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
processed_article = re.sub(r'\s+', ' ', processed_article)

# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

all_sentences

"""## CBOW"""

word2vec = Word2Vec(all_words,sg=0, min_count=10)

vocabulary = word2vec.wv.vocab
print(vocabulary)

sim_words = word2vec.wv.most_similar('coronavirus')

sim_words

"""## Skip-gram"""

word2vec = Word2Vec(all_words,sg=1, min_count=10)

vocabulary = word2vec.wv.vocab
print(vocabulary)

sim_words = word2vec.wv.most_similar('coronavirus')

sim_words