#*------------------------------------------------------------------------------------------------------------------
#Challenge 11 (NLP with Keras-TF2)
 
#Use the yelp review data set available at https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
#Create a NLP text classification model using CNNs and see if you can achieve more than 80% accuracy on TEST data.
 
#Learning task: Fine tune your understanding on loss functions and optimizers using
#https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c
#*------------------------------------------------------------------------------------------------------------------

#Read input file 
import pandas as pd
file='D:\\1.Merene\\NLP\\Challenge 11 -Yelp using CNN\\yelp_labelled.txt'
df = pd.read_csv(file, names=['sentence', 'label'], sep='\t')
#with open (file='D:\\1.Merene\\NLP\\Challenge 11 -Yelp using CNN\\yelp_labelled.txt') as f:
#    data=f.read()

#Create training data set & test data set
from sklearn.model_selection import train_test_split
X = df['sentence'].values
y = df['label'].values
#Loading the data from built in reuters dataset in keras
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.25)
print("Train_data ", X_train.shape)
print("Train_labels ", y_train.shape)
print("Test_data ", X_test.shape)
print("Test_labels ", y_test.shape)

#*------------------------------------------------------------------------------------
#Tokenizing sentences
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

#Adding 1 because of  reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

#Padding to a fixed length
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

#*--------------------------------------------------------------------------------------------

#Build Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(layers.Embedding(vocab_size, 100, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#Fit/Train Model
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

NumEpochs = 50
BatchSize = 256

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
history = model.fit(X_train, y_train, epochs=NumEpochs, batch_size=BatchSize, validation_data=(X_test, y_test),callbacks=[early_stop])

#Evaluate the Model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
#*--------------------------------------------------------------------------------------------
#Plot the accuracy for train data and validation data against the no of epochs.
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
#*--------------------------------------------------------------------------------------------
#Use test data to predict the accuracy % of the model 
from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
#print(confusion_matrix(y_test,predictions))
#*--------------------------------------------------------------------------------------------
