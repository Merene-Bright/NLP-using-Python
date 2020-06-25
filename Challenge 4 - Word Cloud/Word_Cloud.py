#*---------------------------------------------------------------------------------------
#To identify if the new incoming requirement is a match to one of the implemented ones
#*---------------------------------------------------------------------------------------

#Import the libraries required
import codecs
import string
import nltk
import numpy
import pandas as pd
import wordcloud
import spacy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Reading data from the html file
f = codecs.open("D:\\1. Merene\\NLP\Challenge 4\\leaderTalk.html","r","utf-8")
doc=f.read()

#Extracting text from html file
soup = BeautifulSoup(doc)
#Remove all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out
#Extract text
text = soup.get_text()
#Break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
#Break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#Remove blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)

stext=sent_tokenize(text)
#Initialize VADER so we can use it within our Python script
sid = SentimentIntensityAnalyzer()
# Calling the polarity_scores method on sid 
pos_str=''
for sent in stext:
    #sent_text=sent.string.strip()
    scores = sid.polarity_scores(sent)
    if scores['compound'] > 0:
	    pos_str = pos_str+','+sent

#Tokenize the data
def tokenize_txt(in_text):
  return (nltk.tokenize.word_tokenize(in_text))
  
all_tokens = tokenize_txt(text.lower())

#Cleaning text - Removing stopwords and punctuation & converting to lowercase
stopwords_c = set(STOPWORDS)
stopwords_c.update(["favorite","thing","book","will","three","one","immediately","last","night","come","going","good","food","dish","get","really","want","know","tsg","aim","news","network","movie","quote","like"])
def clean_string(in_str):
#  in_str = in_str.lower()
    in_str = ' '.join([word for word in in_str if word not in stopwords_c])
 # in_str = ' '.join([word for word in in_str if len(word)>3])
    in_str = ''.join([word for word in in_str if word not in string.punctuation])
    in_str = ''.join([word for word in in_str if word not in '!"''-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n'])
    return in_str.lower()

usefulWords = clean_string(all_tokens)
#usefulWords = tokenize_txt(usefulWords)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='beige',
        stopwords=stopwords_c,
        max_words=60,
        max_font_size=30, 
        scale=3,
        random_state=1 # random value
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    wordcloud.to_file("D:\\1. Merene\\NLP\Challenge 4\\C_W_Cloud_st.png")

show_wordcloud(usefulWords)
