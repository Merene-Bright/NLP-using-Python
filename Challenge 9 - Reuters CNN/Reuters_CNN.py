#*--------------------------------------------------------------------------------------------
#Challenge 9 (NLP with Deep Learning)
 
#The last challenge (Challenge 8) dealt about achieving a model accuracy of around 80% in the 
#multi class prediction for reuters dataset.
 
#Here are few things we can do now.  Please try to implement them.
#1. See if L1 / L2 regularizers can improve the accuracy.
#2. Separate 10% data from train data and use it as validation set.  
#3. Plot the accuracy for train data and validation data against the no of epochs.
#4. Use test data to predict the accuracy % of the model 
 
#Next let's get into an exciting field....
#Using CNNs in NLP
#http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
#https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
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
from tensorflow.keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,Dense,Flatten,Dropout,Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words=1000
maxlen=250

#Loading the data from built in reuters dataset in keras
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=max_words,test_split=0.25,maxlen=maxlen)
#Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/reuters_word_index.json
#557056/550378 [==============================] - 0s 1us/step
print("Train_data ", X_train.shape)
print("Train_labels ", y_train.shape)
print("Test_data ", X_test.shape)
print("Test_labels ", y_test.shape)

#*--------------------------------------------------------------------------------------------
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

#Pad the training sequences
#pad_type = 'post'
#trunc_type = 'post'
#x_train = pad_sequences(X_train, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
#Pad the testing sequences
#x_test = pad_sequences(X_test, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

#Output the results 
print("\nPadded training shape:", x_train.shape)
print("\nPadded testing shape:",x_test.shape)

#Normalizing/Scaling the Data
#scaler = MinMaxScaler()
#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)

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

#*--------------------------------------------------------------------------------------------
#Separate 10% data from train data and use it as validation set 
subset=round(len(x_train)*0.1)
partial_x_train = x_train[subset:]
partial_y_train = one_hot_train_labels[subset:]
x_val = x_train[:subset]
y_val = one_hot_train_labels[:subset]
print("partial_x_train ", partial_x_train.shape)
print("partial_y_train ", partial_y_train.shape)
print("x_val ", x_val.shape)
print("y_val ", y_val.shape)

#partial_x_train = partial_x_train.reshape(1,len(x_train)-subset, max_words, 1)
#x_val= x_val.reshape(1,subset, max_words, 1)
#*--------------------------------------------------------------------------------------------
#Build Model
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_words))
model.add(Conv1D(64, 5, padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.2))
model.add(Flatten())
#l2 regularizer
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(46, activation='softmax'))

#Fit/Train Model
model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

NumEpochs = 20
BatchSize = 256

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
history = model.fit(partial_x_train, partial_y_train, epochs=NumEpochs, batch_size=BatchSize, validation_data=(x_val, y_val),callbacks=[early_stop])
model.summary()
results = model.evaluate(x_val, y_val)

print("Test Loss ", results[0])
print("Test Accuracy ", results[1]*100)
#*--------------------------------------------------------------------------------------------
#Plot the accuracy for train data and validation data against the no of epochs.
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
#*--------------------------------------------------------------------------------------------
#Use test data to predict the accuracy % of the model 
predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))
#print(confusion_matrix(y_test,predictions))
#*--------------------------------------------------------------------------------------------
