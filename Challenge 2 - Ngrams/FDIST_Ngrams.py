#*---------------------------------------------------------
#To identify if the new incoming requirement is a match 
#to the ones implemented in the past
#*---------------------------------------------------------
#you need to select an appropriate measure to reflect the match percentage. You must include STOPWORDS in this solution.

#Import the libraries required
import nltk
import spacy
import pandas as pd
from nltk.tokenize import word_tokenize

nlp = spacy.load('en',disable=['parser', 'tagger','ner'])
	
#Loading input file
file = pd.read_csv("D:\\1. Merene\\NLP\Challenge 2\\ticket_Data.csv")
file_new = file['Description'].str.lower()

#Writing it into a new file
file_new.to_csv(r'D:\\1. Merene\\NLP\Challenge 2\\ticket_Data_Descr.csv', header=None,index = False)
#Reading data from the file and tokenizing it
with open("D:\\1. Merene\\NLP\Challenge 2\\ticket_Data_Descr.csv") as f:
  doc=f.read()

def tokenize_txt(in_text):
  return (nltk.tokenize.word_tokenize(in_text))
  
all_tokens = tokenize_txt(doc)
print(f"Number of tokens is", len(all_tokens))
#*----------------------------------------------------------------------
#To find ngrams using fdist
#To extract the key frequently occurring ngrams (where n>1 and n<=4) 
#based on n=1 most frequently occurring terms
#*----------------------------------------------------------------------
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords

# remove punctuations from the text
filtered_words1 = ' '.join([w for w in all_tokens if not w in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '])
usefulWords1 = tokenize_txt(filtered_words1)
# remove stopwords from the text
stop_words = set(stopwords.words('english'))
filtered_words = ' '.join([w for w in usefulWords1 if not w in stop_words])
usefulWords = tokenize_txt(filtered_words)

# Get the frequency distribution of the remaining words
fdist = nltk.FreqDist(usefulWords)
count_frame = pd.DataFrame(fdist, index =[0]).T
count_frame.columns = ['Count']

print("\n")		
print(f"Number of tokens is", len(usefulWords))

# Function to get n-grams for a given set of words and frequency
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

# Plot the frequency of the top n n-grams
def n_grams_plot(c_frame,n):
  counts = c_frame.sort_values('Count', ascending = False)
  fig = plt.figure(figsize=(8, 6))
  ax = fig.gca()    
  counts['Count'][:n].plot(kind = 'bar', ax = ax, color='green')
  ax.set_title('Frequency of the most common n-grams')
  ax.set_ylabel('Frequency of n-gram')
  ax.set_xlabel('n-gram')
  plt.show()

# Generating and plotting n-grams from 2 to 4
for i in range(3):
  print(f"********Most common",i+2,"Grams*********")
  r_fc_df=n_grams_freq(usefulWords,i+2)
  n_grams_plot(r_fc_df,10)
