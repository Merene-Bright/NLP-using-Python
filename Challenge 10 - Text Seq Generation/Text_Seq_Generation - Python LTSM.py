#*-------------------------------------------------------------------------------------------------------------
#Challenge 10 : A fun exercise, here you can find a list of dino names. 
#Your objective is to build a sequence model using python or tensorflow that can create new dino names 
#based on the seed input. Please use a character level generation.

#Try building your own LSTM in python.  That will help understanding the different components within a single cell of LSTM.
#*-------------------------------------------------------------------------------------------------------------

#Import the libraries required
import pandas as pd
import numpy as np
#%tensorflow_version 2.x
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Flatten,Dense,LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,LambdaCallback
import random
import sys
import io

#Reading the file
#with open ("D:\\1.Merene\\NLP\Challenge 10 - Dino Names Generation\\dinos.txt") as file:
with open ("/content/sample_data/dinos.txt") as file: #From Collab
    text=file.read().lower()
	
text = text.replace("\n", "!").strip()

#Text Processing
chars = sorted(list(set(text)))
print(len(text))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#Creating 2 arrays : 1 with maxlen characters and the immediate character
maxlen = 1
step = 1
cut_maxlen_chars = []
next_char = []
for i in range(0, len(text) - maxlen, step):
    cut_maxlen_chars.append(text[i: i + maxlen])
    next_char.append(text[i + maxlen])
print('nb sequences:', len(cut_maxlen_chars))

#Vectorization
x = np.zeros((len(cut_maxlen_chars), maxlen, len(chars)), dtype=np.int)
y = np.zeros((len(cut_maxlen_chars), len(chars)), dtype=np.int)
print('Shape of x is',x.shape)
print('Shape of y is',y.shape)
for i, cc in enumerate(cut_maxlen_chars):
    for t, char in enumerate(cc):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_char[i]]] = 1

#Build Model
print('Build model...')
model = Sequential()
model.add(LSTM(400, input_shape=(maxlen, len(chars))))
#model.add(Dropout(0.3))
#model.add(Flatten())
#model.add(Dense(100, activation='relu'))
#model.add(Flatten())
model.add(Dense(len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#Function to generate names
def gen_names(epoch, _,seq_len,seed,r):
    name_len = random.randint(round(seq_len/2), round(seq_len/2)+10)
    eon=0
    for i in range(0,name_len,1):
        while (eon==0): #Stop if end of name has reached
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(seed):
                x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / 1.1
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            next_index = np.argmax(probas)
            next_char = indices_char[next_index]
            if (next_char=='!'):
                eon=1
            else:
                seed = seed[1:] + next_char
                sys.stdout.write(next_char)
            sys.stdout.flush()
    print(seed)

#Generates names at the end of defined number of epochs 
def print_names(epoch, _):
    # Function invoked at end of each epoch
    n=nepoch
    if ((epoch % n==0 and epoch>0) or (epoch==(nepoch-1)) or (est==1)):
      print('Generating names after epoch: %d' % epoch)
      start_index = random.randint(1, 26)
      seed=indices_char[start_index]
      #seed = text[start_index: start_index + maxlen]
      print('Random seed character is : "' + seed + '"')
      for i in range(names):
        gen_names(epoch, _,seq_len,seed,i)

#Fit the model
early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
print_callback = LambdaCallback(on_epoch_end = print_names)

nepoch=500
names=3
seq_len=30
est=0
history = model.fit(x, y,
          batch_size=32,
          epochs=nepoch,
          callbacks=[print_callback,early_stop])

#If early stop, then print the names 
es=len(history.history['loss'])
if (es<nepoch):
   print('Early stopping...')
   est=1
   print_names(es, _)
   