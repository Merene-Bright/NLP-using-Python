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
nlp = spacy.load('en')
tokens = nlp(text)

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
  
all_tokens = tokenize_txt(pos_str.lower())

#Cleaning text - Removing stopwords and punctuation & converting to lowercase
def clean_string(in_str):
#  in_str = in_str.lower()
  stop_words = nltk.corpus.stopwords.words('english')
 # newStopWords= ["favorite","thing","really","want","know","tsg","aim","news","network","movie","quote","like"]
 # stop_words.extend(newStopWords)
  in_str = ' '.join([word for word in in_str if word not in stop_words])
  in_str = ''.join([word for word in in_str if word not in string.punctuation])
  return in_str.lower()

usefulWords = clean_string(all_tokens)
#print(cleaned_text)	
#print(f"Number of tokens is", len(usefulWords))

def n_grams_freq(text_list,n):
  nGramsInDoc = []
  nGrams = ngrams(text_list, n)
  for grams in nGrams:
    nWords = ' '.join(g for g in grams)
    nGramsInDoc.append(nWords)
 # Count the frequency of each n-gram
  fdist = nltk.FreqDist(nGramsInDoc)
  df_frame = pd.DataFrame(fdist, index =[0]).T
  df_frame.columns = ['Count']
  return df_frame

def show_wordcloud(data, title = None):
    from nltk.corpus import stopwords
    stopwords_c = nltk.corpus.stopwords.words('english')
    newStopWords= ["favorite","thing","book","food","dish","get","really","want","know","tsg","aim","news","network","movie","quote","like"]
    stopwords_c.extend(newStopWords)
    wordcloud = WordCloud(
        background_color='beige',
        stopwords=stopwords_c,
        max_words=100,
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

r_fc_df=n_grams_freq(usefulWords,2)
show_wordcloud(usefulWords)
