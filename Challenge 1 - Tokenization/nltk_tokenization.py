#*---------------------------------------------------------
#To tokenize Ticket Description in a csv file using nltk
#*---------------------------------------------------------

#Import the libraries required
import nltk
import pandas
from nltk.tokenize import word_tokenize

#Converting text to lower case and selecting only the 'Description' column
file = pandas.read_csv("D:\\1. Merene\\NLP\Challenge 1\\ticket_Data.csv")
file_new = file['Description'].str.lower()

# remove numeric digits
#txt = ''.join(c for c in file_new if not c.isdigit())
# remove punctuation
#txt = ''.join(c for c in txt if c not in punctuation)

#Writing it into a new file
file_new.to_csv(r'D:\\1. Merene\\NLP\Challenge 1\\ticket_Data_Descr.csv', header=None,index = False)
#Reading data from the file and tokenizing it
with open("D:\\1. Merene\\NLP\Challenge 1\\ticket_Data_Descr.csv") as f:
  doc=f.read()
all_tokens = nltk.word_tokenize(doc)

#Filtering the unique tokens
unique_tokens = set()
result = []
for word in all_tokens:
    if word not in unique_tokens:
        unique_tokens.add(word)
        result.append(word)
		
print(f"Number of tokens is", len(all_tokens))
print(f"Number of unique tokens is", len(unique_tokens))

