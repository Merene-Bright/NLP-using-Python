#---------------------------------------------------------------------------------------
#The Challenge will be to identify the sarcasm in the text. 
#The following is the source of the data https://rishabhmisra.github.io/publications/
#---------------------------------------------------------------------------------------

import numpy as np
import os
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

#Loading data
import pandas as pd
labelled_data='/content/drive/My Drive/Colab Notebooks/imdb_keras_dataset.csv'
test_data ='/content/drive/My Drive/Colab Notebooks/testSentimentDataforValidation.csv'
df_imdb = pd.read_csv(labelled_data) #It already has headers
test_df = pd.read_csv(test_data)

#Displaying the comparison of the texts
df_imdb['Review_len'] = df_imdb.Review.apply(lambda x: len(x.split()))
positive = df_imdb[df_imdb.Label == 1]
negative = df_imdb[df_imdb.Label == 0]

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
sns.distplot(positive.Review_len, hist= True, label= 'Positive')
sns.distplot(negative.Review_len, hist= True, label= 'Negative')
plt.legend()
plt.title('Review distribution by Class', fontsize = 10)
plt.show()

#---------------------------------------------------------------------------------------
#Cleansing the data

import re, string
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

lem = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuations = string.punctuation

def clean_text(text):
    """This function receives sentence and returns clean sentence"""
    text = text.lower()
    text = re.sub("\\n", "", text)
    #text = re.sub("\W+", " ", text)
    
    #Split the sentences into words
    words = list(text.split())
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if w not in punctuations]
    #words = [w for w in words if w not in stop_words]
	#words = [''.join(x for x in w if x.isalpha()) for w in words]

    clean_sen = " ".join(words)
    return clean_sen

df_imdb['imdb_review'] = df_imdb.Review.apply(lambda text: clean_text(text)) 
df_imdb.head()


#---------------------------------------------------------------------------------------
#Creating training & test data

from sklearn.model_selection import train_test_split

reviews = df_imdb['imdb_review']
labels = df_imdb['Label'] 

train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, labels, test_size=0.2, stratify=labels, random_state=42)

max_words = 10000     # how many unique words to use (i.e num rows in embedding vector)
max_len = 500       # max number of words in a headline to use
oov_token = '00_V'    # for the words which are not in training samples
padding_type = 'post'   # padding type
trunc_type = 'post'    # truncation for reviews longer than max length
embed_size = 100    # how big is each word vector

#Tokenization
tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

#Vectorization
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

#---------------------------------------------------------------------------------------
#Building the model

from tensorflow.keras.callbacks import ReduceLROnPlateau 

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

history = model.fit(train_sequences, train_labels, batch_size=32, epochs=4, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)

#---------------------------------------------------------------------------------------
#Evaluating the model	
score1 = model.evaluate(train_sequences, train_labels)
print('Training Loss: ', score1[0])
print('Training Accuracy', score1[1])				
score2 = model.evaluate(test_sequences, test_labels)
print('Test Loss: ', score2[0])
print('Test Accuracy', score2[1])

#All data in history
print(history.history.keys())
#Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#---------------------------------------------------------------------------------------
#Use test data to predict the accuracy % of the model 
from sklearn.metrics import classification_report,confusion_matrix

#Cleansing & formatting text data
test_df['sentences']= test_df['sentences'].str[3:]
test_df['sentences'] = test_df.sentences.apply(lambda text: text.replace("<br />",""))
test_df['reviews'] = test_df.sentences.apply(lambda text: clean_text(text)) 
X_test_pred = test_df['reviews']

tokenizer_t = Tokenizer(num_words=max_words, oov_token=oov_token)
tokenizer_t.fit_on_texts(X_test_pred)
X_test_pred = tokenizer.texts_to_sequences(X_test_pred)
X_test_pred = pad_sequences(X_test_pred, maxlen=max_len, padding=padding_type, truncating=trunc_type)

predictions = model.predict_classes(X_test_pred)
#np.argmax(model.predict(x), axis=-1)
test_df['pred_val']=predictions
test_df.to_csv(r'/content/sample_data/imdb_pred_data.csv')

#*--------------------------------------------------------------------------------------------
#For API
dataSet = test_df[['Id','pred_val']]
#dataSet.to_csv(r'D:\\1.Merene\\NLP\\Hackathon 1 - Sarcasm\\pred_data.csv')
dataSet = dataSet.to_numpy()

dataString = ""
for loop in range(dataSet.shape[0]):
  if loop == 0:
    dataString = str(int(dataSet[loop][0]))+','+str(int(dataSet[loop][1]))
  else:
    dataString = dataString+"\n"+str(int(dataSet[loop][0]))+','+str(int(dataSet[loop][1]))

with open('/content/sample_data/pred_datastring.csv','w') as of:
  of.write(dataString)
  
'''  
postData = {}
postData['challengeName'] = 'sentiment'
postData['userID'] = <<Your Id>>
postData['challengeType'] = 'binaryclassification'
postData['submissionsData'] = dataString

url = 'url'
x = requests.post(url,json=postData)

print(x.text)
#------------------------------------------------------------------------
#To save & load the best model

from tensorflow.keras.models import load_model
# save entire model to HDF5 
model.save("/content/drive/My Drive/Colab Notebooks/imdb_model_best.h5")
# loading the model HDF5
model2 = load_model("/content/drive/My Drive/Colab Notebooks/imdb_model_best.h5")
pred = model2.predict(x)
'''
#*--------------------------------------------------------------------------------------------
