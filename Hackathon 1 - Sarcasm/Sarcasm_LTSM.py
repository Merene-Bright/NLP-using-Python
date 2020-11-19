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
labelled_data='/content/sample_data/training_data.csv'
test_data ='/content/sample_data/test_data.csv'
#labelled_data='D:\\1.Merene\\NLP\\Hackathon 1 - Sarcasm\\training_data.csv'
#test_data ='D:\\1.Merene\\NLP\\Hackathon 1 - Sarcasm\\test_data.csv'
#labelled_df = pd.read_csv(labelled_data, names=['Id','article_link','headline','is_sarcastic'])
df_News = pd.read_csv(labelled_data) #It already has headers
test_df = pd.read_csv(test_data)

#Displaying the comparison of the texts
df_News['headline_len'] = df_News.headline.apply(lambda x: len(x.split()))
sarcastic = df_News[df_News.is_sarcastic == 1]
legit = df_News[df_News.is_sarcastic == 0]

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
sns.distplot(sarcastic.headline_len, hist= True, label= 'Sarcastic')
sns.distplot(legit.headline_len, hist= True, label= 'legitimate')
plt.legend()
plt.title('News Headline Length Distribution by Class', fontsize = 10)
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

df_News = df_News.drop(columns=['article_link'])
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
	

df_News['news_headline'] = df_News.headline.apply(lambda text: clean_text(text)) 
df_News.head()

headlines = df_News['news_headline']
labels = df_News['is_sarcastic'] 

#---------------------------------------------------------------------------------------
#Building the model
#pip install sentencepiece
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow_hub as hub 
import tokenization
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=128):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=128):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
 #   net = tf.keras.layers.Dense(32, activation='relu')(net)
 #   net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

from sklearn.model_selection import train_test_split
max_len = 50
X_train, X_val, y_train, y_val = train_test_split(headlines, labels, test_size=0.1, stratify=labels, random_state=0)
X_train = bert_encode(X_train, tokenizer, max_len=max_len)
X_val = bert_encode(X_val, tokenizer, max_len=max_len)

model = build_model(bert_layer, max_len=max_len)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, verbose=1)

train_history = model.fit(
    X_train, y_train, 
    validation_split=0.1,
    epochs=30,
    callbacks=[checkpoint, earlystopping],
    batch_size=16,
    verbose=1
)

#---------------------------------------------------------------------------------------
#Evaluating the model	
score1 = model.evaluate(X_train, y_train)
print('Training Loss: ', score1[0])
print('Training Accuracy', score1[1])				
score2 = model.evaluate(X_val, y_val)
print('Test Loss: ', score2[0])
print('Test Accuracy', score2[1])

#All data in history
print(train_history.history.keys())
#Summarize history for accuracy
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Summarize history for loss
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#---------------------------------------------------------------------------------------
#Use test data to predict the accuracy % of the model 
from sklearn.metrics import classification_report,confusion_matrix

#Cleansing & formatting text data
test_df = test_df.drop(columns=['article_link'])
test_df['news_headline'] = test_df.headline.apply(lambda news: clean_text(news)) 
X_test_pred = test_df['news_headline']

X_test_pred = bert_encode(X_test_pred, tokenizer, max_len=max_len)

predictions = model.predict(X_test_pred)
#np.argmax(model.predict(x), axis=-1)
test_df['pred_val']=predictions
test_df['pred_val']=test_df['pred_val'].round(decimals=0)
test_df.to_csv(r'/content/sample_data/pred_data.csv')

#*--------------------------------------------------------------------------------------
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
postData['challengeName'] = 'sarcasm'
postData['userID'] = <<Your Employee Id>> 
postData['challengeType'] = 'binaryclassification'
postData['submissionsData'] = dataString

import requests
url = 'https://8n46gxwibi.execute-api.us-east-2.amazonaws.com/default/computeModelScore'
x = requests.post(url,json=postData)

print(x.text)
#------------------------------------------------------------------------
#To save & load the best model

from tensorflow.keras.models import load_model
# save entire model to HDF5 
model_lstm.save("/content/sample_data/model_best.h5")
# loading the model HDF5
model2 = load_model("/content/sample_data/model_best.h5")
pred = model2.predict(x)
'''
#*--------------------------------------------------------------------------------------------
