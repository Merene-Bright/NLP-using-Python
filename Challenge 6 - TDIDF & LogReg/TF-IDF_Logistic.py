#*--------------------------------------------------------------------------------------------
#Challenge 6 :  This is an extension to the Challenge 5 .  Assume you are looking to build 
#a model to predict if the text you are seeing is an issue or informational message. 
#For e.g. job completed is an informational message whereas job abended is an issue.  
#The objective is to predict the issue.
 
#Build a TF-IDF based matrix along with the label
#Explore giving that input into a logistic regression to predict if the given text is an issue
 
#Do you see an improvement over Challenge 5 implementation?
#*--------------------------------------------------------------------------------------------

#Import the libraries required
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

#Loading input file
df = pd.read_csv("D:\\1. Merene\\NLP\Challenge 5\\ticket_Data.csv")
df.head()

# REMOVE NaN VALUES AND EMPTY STRINGS:
df.dropna(inplace=True)

blanks = []  # start with an empty list
for i,tid,dsc in df.itertuples():  # iterate over the DataFrame
    if type(dsc)==str:            # avoid NaN values
        if dsc.isspace():         # test 'description' for whitespace
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
df[['Description','Label']].to_csv('D:\\1. Merene\\NLP\Challenge 6\\TicketData_Labelled.csv')

#Reading data from the file
rdf = pd.read_csv('D:\\1. Merene\\NLP\Challenge 6\\TicketData_Labelled.csv',usecols=['Description','Label'])
rdf.head()

#Split the data into train & test sets
X = rdf['Description']  # Looking at the text
y = rdf['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#Combine Steps with TfidVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train) # remember to use the original X_train set
X_train_tfidf.shape

#Train an LR classifier
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train_tfidf, y_train)

#Build a Pipeline
stopwords_c = set(stopwords.words('english'))
text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords_c)),('clf', LogisticRegression(solver='lbfgs')),])

#Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  

#Form a prediction set
predictions = text_clf.predict(X_test)

#Report the confusion matrix
print(f"Confusion Matrix:\n",metrics.confusion_matrix(y_test,predictions))

#Print a classification report
print(f"\nClassification Report:\n",metrics.classification_report(y_test,predictions))

#Print the overall accuracy
print(f"\nAccuracy:",metrics.accuracy_score(y_test,predictions))
