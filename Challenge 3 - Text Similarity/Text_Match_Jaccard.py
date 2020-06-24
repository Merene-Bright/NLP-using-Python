#*---------------------------------------------------------------------------------------
#To identify if the new incoming requirement is a match to one of the implemented ones
#*---------------------------------------------------------------------------------------

#Import the libraries required
import pandas as pd
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

#Reading data from the file
requirements = pd.read_csv(("D:\\1. Merene\\NLP\Challenge 3\\Input_RFP - Requirements.csv"), engine='python', usecols=['Requirement'])

#Cleaning text - Removing stopwords and punctuation & converting to lowercase
def clean_string(in_str):
  in_str = in_str.lower()
  stop_words = set(stopwords.words('english'))
  in_str = ' '.join([word for word in in_str.split() if word not in stop_words])
  in_str = ''.join([word for word in in_str if word not in string.punctuation])
  return in_str

cleaned_text = list(map(clean_string,requirements['Requirement']))
#print(cleaned_text)

#Reading new input requirement to check against the implemented ones
in_requirement = pd.read_csv(("D:\\1. Merene\\NLP\Challenge 3\\Test_Input.csv"), engine='python', usecols=['Requirement'])
print(f'The input requirement is "', in_requirement['Requirement'][0], '"','\n')
in_cleaned_text = list(map(clean_string,in_requirement['Requirement']))

#key_requirement = input("Enter new requirement:")
#print(f'The input requirement is "', key_requirement, '"','\n')
#in_cleaned_text = list(map(clean_string,key_requirement))

#Jaccard's Similarity function
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

s_max=0
js =0
index_a=[]
js_a=[]
for i in range(len(cleaned_text)):
    #js = jaccard_similarity(in_cleaned_text[0],cleaned_text[i])
    js = jaccard_similarity(in_cleaned_text[0],cleaned_text[i])
    #print(s_max,i,js)
    index_a.append(i)
    js_a.append(js)
   
max_js=max(js_a)

if max_js == 0 :
    print(f'There is no similar requirement which has been implemented.')
	#break
else:
    for i in range(len(js_a)):
        if js_a[i] == max_js:
            print(f'The MOST similar requirement already implemented is "', requirements['Requirement'][i], '"','\n')
            print(f'The Jaccard similarity is ', js_a[i])

		#A totally new requirement
