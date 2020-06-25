#*--------------------------------------------------------------------------------------------
#1.To build a model to predict if the text you are seeing is an issue or informational message
#2. Generate the label for each data point to see if its informational or an issue
#3. Build a count based matrix for each issue which reflects the number of occurrence 
#   of each word in the text along with the label
#->You can choose to restrict to a certain number of words based on your analysis
#->You can choose to remove the STOPWORDS from each text 
#5. The count based matrix that you build for the entire data set should be in a format such 
#   that it can be given as an input to any of the algorithm for training and testing.
#*--------------------------------------------------------------------------------------------

#Import the libraries required
import nltk
import string
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist

#Loading input file
csvfile = open("D:\\1. Merene\\NLP\Challenge 5\\ticket_Data.csv",'r')
reader = csv.reader(csvfile)

def tokenize_txt(in_text):
  return (nltk.tokenize.word_tokenize(in_text))
  
#Removing stopwords and punctuation & converting to lowercase
stopwords_c = set(stopwords.words('english'))
stopwords_c.update(["got"])
def clean_string(in_str):
    in_str = ' '.join([word for word in in_str if word not in stopwords_c])
    in_str = ''.join([word for word in in_str if word not in string.punctuation])
 #   in_str = ''.join([word for word in in_str if word not in '!"''-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n'])
    return in_str.lower()

sid = SentimentIntensityAnalyzer()
#neg_str= [[i * j for j in range(len(list(myreader)))] for i in range(2)]
neg_str=[]
pn_str=[]

f = open("D:\\1. Merene\\NLP\Challenge 5\\ticket_Data_Neg_Word_Count.csv", 'w', newline='')
filewriter=csv.writer(f)
filewriter.writerow(['Sentence','Word','Count','Label'])
f.close()
for row in reader:
   # print(row[1])
    stext = sent_tokenize(row[1])
    #print(stext)
   # Calling the polarity_scores method on sid 
    for sent in [line.strip() for line in stext if line.strip() != ""]:
        scores = sid.polarity_scores(sent)
        #If it is negative, it is considered as issue
        if scores['compound'] < 0:
            neg_str.append(sent)
            neg_token_text=tokenize_txt(sent)
            neg_clean_text=clean_string(neg_token_text)
            neg_token_text=tokenize_txt(neg_clean_text)
            neg_tokens.append(neg_token_text)
            neg_fdist = nltk.FreqDist(neg_token_text)
            with open('D:\\1. Merene\\NLP\Challenge 5\\ticket_Data_Neg_Word_Count.csv', mode='a+', newline='') as neg_words:
                for word in neg_fdist:
                    filewriter=csv.writer(neg_words)
                    filewriter.writerow([sent,word,neg_fdist[word],'0'])
                neg_words.close()
        #If it is positive or neutral, it is considered as information
        else:
            pn_str.append(sent)

f = open("D:\\1. Merene\\NLP\Challenge 5\\ticket_Data_labelled_input.csv", 'w', newline='')
filewriter=csv.writer(f)
filewriter.writerow(['Sentence','Label'])
f.close()
with open("D:\\1. Merene\\NLP\Challenge 5\\ticket_Data_labelled_input.csv", 'a+', newline='') as neg_data:
    for i in range(len(neg_str)):
        filewriter=csv.writer(neg_data)
        filewriter.writerow([neg_str[i],'0'])
    neg_data.close()
with open("D:\\1. Merene\\NLP\Challenge 5\\ticket_Data_labelled_input.csv", 'a+', newline='') as pn_data:
    for i in range(len(pn_str)):
        filewriter=csv.writer(pn_data)
        filewriter.writerow([pn_str[i],'1'])
    pn_data.close()
 
csvfile.close()
