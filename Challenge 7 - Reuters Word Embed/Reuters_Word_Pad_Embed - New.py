#*------------------------------------------------------------------------------------------------------
#Challenge 7 :  Implement the following on the built in reuters dataset in keras

#1. Tokenization 
#2. Padding
#3. Wordindex creation and reverse word index
#4. Embedding layer for word embeddings
 
#Some useful links below:
#a. Tokenizaton and Padding - What are they and how to use keras to accomplish them?
#https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html
#https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html
 
#b. Word index and Reverse word index using keras - how to getback words from word indices? 
#https://stackoverflow.com/questions/41971587/how-to-convert-predicted-sequence-back-to-text-in-keras
 
#c. How to use the power of word embeddings - one of the greatest things in NLP - 
#They provide representation of words and their relative meanings.
#https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#*------------------------------------------------------------------------------------------------------

#Import the libraries required
import tensorflow
from tensorflow import keras
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models,layers,regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import preprocessing
#from tensorflow.keras.utils import np_utils
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report,confusion_matrix

# Number of words to consider as features
max_words = 8000
# Cut texts after this number of words 
maxlen = 250

#Loading the data from built in Reuters dataset in keras
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=max_words,test_split=0.3, maxlen=maxlen)
print("Train_data ", X_train.shape)
print("Train_labels ", y_train.shape)
print("Test_data ", X_test.shape)
print("Test_labels ", y_test.shape)

#This dataset also makes available the word index used for encoding the sequences:
#Note there are 30979 words (will be used for our embedding)
word_index = reuters.get_word_index(path="reuters_word_index.json")

#********************************************************************************************************
#Building reverse data
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#Building text for first x articles from Training data
x=len(X_train)
#x=1000
decoded_newswire=''
train_all_newswire=[]
for i in range(x):
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in X_train[i]])
    train_all_newswire.append(decoded_newswire)

#Building text for first y articles from Test data
y=len(X_test)
#y=150
decoded_newswire=''
test_all_newswire=[]
for i in range(y):
    decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in X_test[i]])
    test_all_newswire.append(decoded_newswire)

#********************************************************************************************************
pad_type = 'post'
trunc_type = 'post'

def tokenize_txt(in_text):
    return (nltk.tokenize.word_tokenize(in_text))

#all_train_tokens =[]
#for i in range(len(train_all_newswire)):
#    all_train_tokens.append(tokenize_txt(train_all_newswire[i]))

stop_words = set(stopwords.words('english'))
def clean_string(in_str):
#  in_str = in_str.lower()
    in_str = ' '.join([word for word in in_str if word not in stop_words])
 # in_str = ' '.join([word for word in in_str if len(word)>3])
    in_str = ''.join([word for word in in_str if word not in string.punctuation])
    in_str = ''.join([word for word in in_str if word not in '!"''-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n'])
    return in_str.lower()
	
#Bag of words
vocab = {}
i = 1
header=[]
train_all_newswire_n=[]
for row in range(len(train_all_newswire)):  # iterate over the DataFrame
    x = train_all_newswire[row].split()        
    x = clean_string(x)
    x = tokenize_txt(x)
    j=0
    train_all_newswire_n.append([0]*maxlen)
    for word in x:
        if word in vocab:
            v=vocab[word]
            train_all_newswire_n[row][j]=v
        else:
            vocab[word]=i
           # print(row,j,i)
            train_all_newswire_n[row][j]=i
            i+=1
            header.append(word)
        j=j+1
    
test_all_newswire_n=[]
for row in range(len(test_all_newswire)):  # iterate over the DataFrame
    x = test_all_newswire[row].split()        
    x = clean_string(x)
    x = tokenize_txt(x)
    j=0
    test_all_newswire_n.append([0]*maxlen)
    for word in x:
        if word in vocab:
            v=vocab[word]
            test_all_newswire_n[row][j]=v
        else:
            vocab[word]=i
            test_all_newswire_n[row][j]=i
            i+=1
            header.append(word)
        j=j+1
print(len(vocab))

#Pad the training sequences
train_padded = pad_sequences(train_all_newswire_n, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
#Pad the testing sequences
test_padded = pad_sequences(test_all_newswire_n, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

#Output the results 
print("\nPadded training shape:", train_padded.shape)
print("\nPadded testing shape:",test_padded.shape)

#ONE HOT ENCODER of the labels
one_hot_train_labels = to_categorical(y_train,46)
one_hot_test_labels = to_categorical(y_test,46)
print("one_hot_train_labels ", one_hot_train_labels.shape)
print("one_hot_test_labels ", one_hot_test_labels.shape)

#********************************************************************************************************
#Build Model
model = models.Sequential()
#model.add(Embedding(len(word_index),8,input_length=maxlen))
model.add(Embedding(max_words, 16, input_length=maxlen))
model.add(layers.Dropout(0.5))
model.add(Flatten())
#model.add(layers.Dense(256, kernel_regularizer=regularizers.l1(0.001), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(Dense(46, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

#Fit/Train Model
NumEpochs = 50
BatchSize = 32
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
history = model.fit(train_padded,one_hot_train_labels, epochs=NumEpochs, batch_size=BatchSize, validation_data=(test_padded, one_hot_test_labels),callbacks=[early_stop])
results = model.evaluate(test_padded, one_hot_test_labels)
print('Test loss:', results[0])
print('Test accuracy:', results[1]*100)

#Loss & Accuracy curves
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

#********************************************************************************************************