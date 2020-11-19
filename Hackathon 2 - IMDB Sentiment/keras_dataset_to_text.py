#--------------------------------------------------------------------------------------------------------------
#The Challenge will be to identify the sentiment in the text in the imdb_reviews. 
#The data imdb_reviews for the training has to be taken from tensorflow tfds.
#85%
#--------------------------------------------------------------------------------------------------------------
#Load data
#!pip install -q tensorflow-hub
#!pip install -q tensorflow-datasets
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers

from keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data()

data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
#Converting to DataFrame
#df_X=pd.DataFrame(data)
#df_X.columns=['Review']
df_y = pd.DataFrame(targets)
df_y.columns=['Label']
#df = pd.concat([df_X,df_y], axis=1)
#df.columns=['Review','Label']

index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()]) 

df_y['Review']=''
for x in range(len(df_y)):
  if(x % 10000==0):
      print('Review number:',x)
  df_y['Review'][x] = " ".join( [reverse_index.get(i - 3, "#") for i in data[x]] )
  #print(decoded) 

print('Processing complete...')
df_y.to_csv(r'/content/sample_data/imdb_keras_dataset.csv')
df_y.to_csv(r'/content/drive/My Drive/Colab Notebooks/imdb_keras_dataset.csv')
print('Writing into the file complete...')


