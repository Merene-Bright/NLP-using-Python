#*------------------------------------------------------------------------------------------------------------------
#Challenge 13 (NLP with Keras-TF2)
#Try to predict spam emails using RNNs (LSTM) in Keras. See if it improves the performance of normal MLPs in Keras. 
 
#Go through this link for a refresher on RNNs #https://pathmind.com/wiki/lstm
#*------------------------------------------------------------------------------------------------------------------

import numpy as np
import warnings
warnings.filterwarnings("ignore")

#Loading input data
import pandas as pd
#file='D:\\1.Merene\\NLP\\Challenge 13 - Classify spam\\datasets_spam.csv'
file='/content/sample_data/datasets_spam.csv'
X, y = [], []
with open(file,encoding = "ISO-8859-1") as f:
    for line in f:
        split = [x.strip() for x in line.split(',')]
        y.append(split[0].strip())
        X.append(' '.join(split[1:]).strip())

#Removing header
X=X[1:]
y=y[1:]

#To convert labels to integers
label2int = {"ham": 0, "spam": 1}
y = [ label2int[label.strip('\"')] for label in y ]

#Converting to DataFrame
df_X=pd.DataFrame(X)
df_y=pd.DataFrame(y)
df = pd.concat([df_X,df_y], axis=1)
df.columns=['SMSMail','is_spam']
df.head()

#Displaying the length of the messages
df['content_len'] = df.SMSMail.apply(lambda x: len(x.split()))
spam = df[df.is_spam == 0]
ham = df[df.is_spam == 1]

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
sns.distplot(spam.content_len, hist= True, label= 'Spam')
sns.distplot(ham.content_len, hist= True, label= 'Ham')
plt.legend()
plt.title('SMS/Email Length Distribution by Class', fontsize = 10)
plt.show()

#--------------------------------------------------------------------------
#Cleansing the message content

import re, string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

lem = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
punctuations = string.punctuation

def clean_text(text):
    """This function receives returns clean text"""
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
	
df['message'] = df.SMSMail.apply(lambda mail: clean_text(mail)) 
df.head()

#--------------------------------------------------------------------------
#Creating training data and test data
from sklearn.model_selection import train_test_split
content = df['message']
labels = df['is_spam'] 

train_sentences, test_sentences, train_labels, test_labels = train_test_split(content, labels, test_size=0.2, stratify=labels, random_state=42)

max_words = 30000     # how many unique words to use (i.e num rows in embedding vector)
max_len = 70       # max number of words in a message to use
oov_token = '00_V'    # for the words which are not in training samples
padding_type = 'post'   # padding type
trunc_type = 'post'    # truncation for headlines longer than max length
embed_size = 100    # how big is each word vector

tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

#--------------------------------------------------------------------------
#Building the model

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau 

model_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, embed_size, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.2),
    tf.keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)
model_lstm.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model_lstm.summary()

history_lstm = model_lstm.fit(train_sequences, train_labels, batch_size=32, epochs=4, 
                    validation_data=(test_sequences, test_labels), 
                    callbacks=[rlrp] ,verbose=1)

#--------------------------------------------------------------------------
#Evaluating the model performance

score = model_lstm.evaluate(train_sequences, train_labels)
print('Training Loss: ', score[0])
print('Training Accuracy', score[1])
					
score = model_lstm.evaluate(test_sequences, test_labels)
print('Test Loss: ', score[0])
print('Test Accuracy', score[1])
	
#Print all data in history
print(history_lstm.history.keys())

#Summarize history for accuracy
plt.plot(history_lstm.history['accuracy'])
plt.plot(history_lstm.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Summarize history for loss
plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#--------------------------------------------------------------------------
