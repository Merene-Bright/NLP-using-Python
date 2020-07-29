#*-------------------------------------------------------------------------------------------------------------------
#Challenge 14 (NLP with Keras-TF2)
 
#Learn how to generate text using LSTM and use it with your favorite ebook to generate some meaningful sentences.
#https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms
#*-------------------------------------------------------------------------------------------------------------------

#Reading the data 
#with open ("D:\\1.Merene\\NLP\Challenge 14 - Dino Names Generation\\sherlock_homes.txt") as file:
with open('/content/sample_data/Sherlock Holmes.txt', encoding = "ISO-8859-1") as file:
    text = file.read().lower()

#*-------------------------------------------------------------------------------------------------------------------
#Text processing
chars = sorted(list(set(text))) # getting all unique chars
print('Text length', len(text))
print('Total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#Creating 2 arrays : 1 with maxlen characters and the immediate character
import numpy as np
maxlen = 40
step = 3
sentences  = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences .append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Sequences:', len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.int)
y = np.zeros((len(sentences), len(chars)), dtype=np.int)
print('Shape of x is',x.shape)
print('Shape of y is',y.shape)
#Vectorization
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
	
#*-------------------------------------------------------------------------------------------------------------------
#Build Model
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from keras.optimizers import RMSprop

model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
#optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#*-------------------------------------------------------------------------------------------------------------------
#Helper function from Keras team
import random

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#*-------------------------------------------------------------------------------------------------------------------
#Defining callbacks
from keras.callbacks import EarlyStopping,LambdaCallback,ModelCheckpoint,ReduceLROnPlateau

filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=1, min_lr=0.001)
callbacks = [checkpoint, reduce_lr]		
#*-------------------------------------------------------------------------------------------------------------------				 
#Train & Fit the model
import sys
history=model.fit(x, y, batch_size=128, epochs=10, callbacks=callbacks)
#*-------------------------------------------------------------------------------------------------------------------				 
#Function to generate text based on helper function from Keras team
def generate_text(length, diversity):
    # Get random starting text
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
          x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated
	
print(generate_text(400, 0.35))
#*-------------------------------------------------------------------------------------------------------------------	
