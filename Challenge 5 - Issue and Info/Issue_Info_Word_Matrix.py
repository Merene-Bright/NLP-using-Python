#*--------------------------------------------------------------------------------------------
#This is an extension to the Challenge 2  .  Assume you are looking to build a model to 
#predict if the text you are seeing is an issue or informational message. 
#For e.g. job completed is an informational message whereas job abended is an issue.  
#The objective is to predict the issue.
#
#The task is multi fold
#1. Generate the label for each data point to see if its informational or an issue
#2. Build a count based matrix for each issue which reflects the number of occurrence of 
#3. each word in the text along with the label
#->You can choose to restrict to a certain number of words based on your analysis
#->You can choose to remove the STOPWORDS from each text 
#The count based matrix that you build for the entire data set should be in a format such 
#that it can be given as an input to any of the algorithm for training and testing.
#*--------------------------------------------------------------------------------------------

#Import the libraries required
import numpy as np
import pandas as pd
import nltk
import string
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

#Loading input file
df = pd.read_csv("D:\\1. Merene\\NLP\Challenge 5\\ticket_Data.csv")
df.head()

# REMOVE NaN VALUES AND EMPTY STRINGS:
df.dropna(inplace=True)

blanks = []  # start with an empty list
#for i,tid,dsc in df.itertuples():  # iterate over the DataFrame
#    if type(dsc)==str:            # avoid NaN values
#        if dsc.isspace():         # test 'description' for whitespace
#            blanks.append(i)     # add matching index numbers to the list
for row in df.itertuples():  # iterate over the DataFrame
    if type(row.Description)==str:            # avoid NaN values
        if row.Description.isspace():         # test 'description' for whitespace
            blanks.append(i)     # add matching index numbers to the list
df.drop(blanks, inplace=True)

sid = SentimentIntensityAnalyzer()
sid.polarity_scores(df.loc[0]['Description'])
#Adding Scores and Labels to the DataFrame
df['Scores'] = df['Description'].apply(lambda Description: sid.polarity_scores(Description))
df['Compound']  = df['Scores'].apply(lambda score_dict: score_dict['compound'])
df['Label'] = df['Compound'].apply(lambda c: 'pos' if c >=0 else 'neg')
df.head()

#Wrting data into a csv file
df[['Description','Label']].to_csv('D:\\1. Merene\\NLP\Challenge 5\\TicketData_Labelled.csv')

#Reading data from the file
rdf = pd.read_csv('D:\\1. Merene\\NLP\Challenge 5\\TicketData_Labelled.csv',usecols=['Description','Label'])
#rdf=idf.loc[idf['Label']== 'neg']
rdf.head()

#Removing stopwords and punctuation & converting to lowercase
stopwords_c = set(stopwords.words('english'))
stopwords_c.update(["got","ws","news","mloa","tdata","tbload.it","r","jobp"])
def clean_string(in_str):
    s_stemmer = SnowballStemmer(language='english')
    p_stemmer = PorterStemmer()
  #  in_str = ' '.join([s_stemmer.stem(word) for word in in_str if word not in stopwords_c])
    in_str = ' '.join([word for word in in_str if word not in stopwords_c])
    in_str = ''.join([word for word in in_str if word not in string.punctuation])
   # in_str = ''.join([word for word in in_str if word not in '!"''-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n'])
    return in_str.lower()

def tokenize_txt(in_text):
    return (nltk.tokenize.word_tokenize(in_text))

#Bag of words
vocab = {}
i = 1
dict_of_df = []
header=['Ticket_Desc','Label']
for row in rdf.itertuples():  # iterate over the DataFrame
    x = row.Description.lower().split()        
    x = clean_string(x)
    x = tokenize_txt(x)
    for word in x:
        if word in vocab:
            continue
        else:
            vocab[word]=i
            i+=1
            header.append(word)
print(len(vocab))

dict_of_df = []
for row in rdf.itertuples():
    dict_of_df.append([row.Description]+[row.Label]+[0]*len(vocab))

#Map the frequencies of each word to our vector:
j = 0
for row in rdf.itertuples():  # iterate over the DataFrame
    if (j > len(dict_of_df)-1):
            break
    x = row.Description.lower().split()         # avoid NaN values
    x = clean_string(x)
    x = tokenize_txt(x)
    for word in x:
        v = -1
        v=vocab[word]
        dict_of_df[j][v+1]=dict_of_df[j][v+1]+1
    j+=1
    
#Wrting data into a file
file_data = open('D:\\1. Merene\\NLP\Challenge 5\\Ticket_Data_Word_Matrix.csv','w')
i=0
for i in range(len(header)):
    if(i==0):
        hstr=header[i]
    else:
        hstr=hstr+','+header[i]
file_data.write(hstr)
file_data.write('\n')
n=0
for n in range(len(dict_of_df)):
    #ij_str=dict_of_df[n][0]
    ij_str=''
    j=1
    for j in range(len(dict_of_df[n])):
        if(j==0):
            ij_str='"'+str(dict_of_df[n][j])+'"'
        else:
            ij_str=ij_str+','+str(dict_of_df[n][j])
    file_data.write(ij_str)
    file_data.write('\n')
file_data.close()