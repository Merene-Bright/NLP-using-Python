#*--------------------------------------------------------------------------------------------
#Challenge 8 :  Implement the following on the built in reuters dataset in keras

#1. Vectorization and Normalization
#2. One Hot encoding for labels using the built in to_categorical util
#3. Use these layers to build the model - Sequential, Dense, Dropout
#4. Achieve a model accuracy of around 82%
#*--------------------------------------------------------------------------------------------

#Import the libraries required
import tensorflow 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.datasets import reuters
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models,layers,regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix

max_words=10000

#Loading the data from built in reuters dataset in keras
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=max_words,test_split=0.25)
#Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters_word_index.json
#557056/550378 [==============================] - 0s 1us/step
print("Train_data ", X_train.shape)
print("Train_labels ", y_train.shape)
print("Test_data ", X_test.shape)
print("Test_labels ", y_test.shape)

def vectorize_sequences(sequences, dimension=max_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #print(i,'+',sequence)
        results[i, sequence] = 1.
    return results
	
# Vectorize train and test to tensors with 10k columns
x_train = vectorize_sequences(X_train)
x_test = vectorize_sequences(X_test)
print("x_train ", x_train.shape)
print("x_test ", x_test.shape)

#Normalizing/Scaling the Data
scaler = MinMaxScaler()
scaler.fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

#ONE HOT ENCODER of the labels
one_hot_train_labels = to_one_hot(y_train)
one_hot_test_labels = to_one_hot(y_test)
#one_hot_train_labels = to_categorical(y_train)
#one_hot_test_labels = to_categorical(y_test)
print("one_hot_train_labels ", one_hot_train_labels.shape)
print("one_hot_test_labels ", one_hot_test_labels.shape)

#Build Model
model = models.Sequential()
model.add(layers.Dense(900, activation='relu', input_shape=(max_words,)))
model.add(layers.Dropout(0.2))
#model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(46, activation='softmax'))
model.summary()

#Fit/Train Model
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

NumEpochs = 4
BatchSize = 256

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
history = model.fit(X_train, one_hot_train_labels, epochs=NumEpochs, batch_size=BatchSize, validation_data=(X_test, one_hot_test_labels),callbacks=[early_stop])
#history = model.fit(X_train, one_hot_train_labels, epochs=NumEpochs, batch_size=BatchSize, validation_data=(X_test, one_hot_test_labels))
results = model.evaluate(X_test, one_hot_test_labels)
print("Test Loss ", results[0])
print("Test Accuracy ", results[1]*100)
history_dict = history.history
history_dict.keys()

#Loss & Accuracy curves
model_loss = pd.DataFrame(model.history.history)
#model_loss.plot()

