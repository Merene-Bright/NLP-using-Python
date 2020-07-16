#*------------------------------------------------------------------------------------------------------------------
#Challenge 12 (NLP with Keras-TF2)
#This is an extension to Challenge 11 to improve the results further through Hyper-parameter optimization
#Use KerasClassifier (to use k-fold cross validation etc.) and RandomizedSearchCV 
#(to find the best combination of Hyper-parameters) and fine tune the model.
#*------------------------------------------------------------------------------------------------------------------

#Read input file 
import pandas as pd
#file='D:\\1.Merene\\NLP\\Challenge 12\\yelp_labelled.txt'
file='/content/sample_data/yelp_labelled.txt'
df = pd.read_csv(file, names=['sentence', 'label'], sep='\t')
#with open (file='D:\\1.Merene\\NLP\\Challenge 11 -Yelp using CNN\\yelp_labelled.txt') as f:
#    data=f.read()

#Create training data set & test data set
from sklearn.model_selection import train_test_split
X = df['sentence'].values
y = df['label'].values

#Splitting the data
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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return model

epochs = 20
embedding_dim = 50
maxlen = 100
batch_size=10

param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7],
                  vocab_size=[5000], 
                  embedding_dim=[50],
                  maxlen=[100])
                  
model = KerasClassifier(build_fn=create_model,epochs=epochs,batch_size=batch_size,verbose=False)

cv=4 #kfold cross validation
grid = RandomizedSearchCV(estimator=model,param_distributions=param_grid,cv=cv, verbose=1, n_iter=5)

#Fit/Train Model
grid_result = grid.fit(X_train, y_train)

#Evaluate the Model using test data
test_accuracy = grid.score(X_test, y_test)

#Publishing the results
print('Results are as follows...')
print('Best score', grid_result.best_score_)
print('Best parameters:\n')
print(grid_result.best_params_)
print('Test accuracy', test_accuracy)

#*--------------------------------------------------------------------------------------------
