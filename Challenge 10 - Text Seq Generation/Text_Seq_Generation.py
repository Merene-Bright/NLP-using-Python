#*-------------------------------------------------------------------------------------------------------------
#Challenge 10 - Challenge 10 : A fun exercise, here you can find a list of dino names. 
#Your objective is to build a sequence model using python or tensorflow that can create new dino names 
#based on the seed input. Please use a character level generation.
#*-------------------------------------------------------------------------------------------------------------

#Import the libraries required
import nltk
import pandas
%tensorflow_version 2.x
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

#Reading the file
with open ("D:\\1.Merene\\NLP\Challenge 10 - Dino Names Generation\\dinos.txt") as file:
    text=file.read()

# The unique characters in the file
vocab = sorted(set(text))
#print(vocab)
#len(vocab)

#Step 2 : Text Processing
#Text Vectorization
#Neural network can't take in the raw string data, we need to assign numbers to each character. 
#Two dictionaries that can go from numeric index to character and character to numeric index.

#Char assigned numbers
char_to_ind = {u:i for i, u in enumerate(vocab)}
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in text])

#Step 3: Creating Batches
seq_len = 80
total_num_seq = len(text)//(seq_len+1)
# Create Training Sequences
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

#for i in char_dataset.take(500):
#     print(ind_to_char[i.numpy()])
	 
sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt
	
dataset = sequences.map(create_seq_targets)

for input_txt, target_txt in  dataset.take(1):
    print(input_txt.numpy())
    print(''.join(ind_to_char[input_txt.numpy()]))
    print('\n')
    print(target_txt.numpy())
    # There is an extra whitespace!
    print(''.join(ind_to_char[target_txt.numpy()]))

# Batch size
batch_size = 100
#Buffer size to shuffle the dataset so it doesn't attempt to shuffle
#the entire sequence in memory. Instead, it maintains a buffer in which it shuffles elements
buffer_size = 1000
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab) #53
# The embedding dimension
embed_dim = 36 #changed from 64
# Number of RNN units
rnn_neurons = 513 #changed from 1026

def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
 
def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size,embed_dim,batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    # Final Dense Layer to Predict
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss) 

    #model.add(LSTM(256, input_shape=(vocab_size,embed_dim), return_sequences=True))
    #model.add(Dropout(0.3))
    #model.add(LSTM(256))
    #model.add(Dense(vocab_size, activation="softmax"))
    return model

model = create_model(vocab_size = vocab_size,embed_dim=embed_dim,rnn_neurons=rnn_neurons,batch_size=batch_size)

model.summary()

for input_example_batch, target_example_batch in dataset.take(1):

  # Predict off some random batch
  example_batch_predictions = model(input_example_batch)
  # Display the dimensions of the predictions
  print(example_batch_predictions.shape, " <=== (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# Reformat to not be a lists of lists
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
print("Given the input seq: \n")
print("".join(ind_to_char[input_example_batch[0]]))
print('\n')
print("Next Char Predictions: \n")
print("".join(ind_to_char[sampled_indices ]))

epochs = 200
early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
model.fit(dataset,epochs=epochs,callbacks=[early_stop])

model.save('dinos_gen.h5') 
from tensorflow.keras.models import load_model
model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
model.load_weights('dinos_gen.h5')
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, start_seed,gen_size=100,temp=1.0):
  '''
  model: Trained Model to Generate Text
  start_seed: Intial Seed text in string form
  gen_size: Number of characters to generate

  Basic idea behind this function is to take in some seed text, format it so
  that it is in the correct shape for our network, then loop the sequence as
  we keep adding our own predicted characters. Similar to our work in the RNN
  time series problems.
  '''
  # Number of characters to generate
  num_generate = gen_size
  # Vecotrizing starting seed text
  input_eval = [char_to_ind[s] for s in start_seed]
  # Expand to match batch format shape
  input_eval = tf.expand_dims(input_eval, 0)
  # Empty list to hold resulting generated text
  text_generated = []

  # Temperature effects randomness in our resulting text
  # The term is derived from entropy/thermodynamics.
  # The temperature is used to effect probability of next characters.
  # Higher probability == lesss surprising/ more expected
  # Lower temperature == more surprising / less expected
 
  temperature = temp
  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):

      # Generate Predictions
      predictions = model(input_eval)
      # Remove the batch shape dimension
      predictions = tf.squeeze(predictions, 0)
      # Use a cateogircal disitribution to select the next character
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # Pass the predicted charracter for the next input
      input_eval = tf.expand_dims([predicted_id], 0)
      # Transform back to character letter
      text_generated.append(ind_to_char[predicted_id])
  return (start_seed + ''.join(text_generated))
  
print(generate_text(model,"Meroktenos",gen_size=50))
