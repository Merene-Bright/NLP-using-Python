Challenge 1: Tokenization : You have a collection of tickets. Apply tokenization using python nltk and generate a vocabulary of words in the ticket. 

Challenge 2 :  Tokenization, FDIST, ngrams This is an extension to the Challenge 1  .  Once you have tokenized, find the n-grams using FDIST function. Extract the key frequently occurring ngrams (where n>1 and n<=4) based on n=1 most frequently occurring terms. Please feel free to explore the inclusion of STOPWORDS in this solution

Challenge 3 :  A Customer is looking for a mechanism to identify if the new incoming requirement is a match to one of the implemented ones.  Enclosed is a sample set of requirements, come up with an approach to identify the match against the incoming requirement, you need to select an appropriate measure to reflect the match percentage. You must include STOPWORDS in this solution.

Challenge 4 :  You have access to a html file called leaderTalk.html which has leaders talk about different dimensions both personal and official, including why they think Cognizant is their favorite organization. The objective is to parse the content related to why they think Cognizant is their favorite organization and generate word cloud that outlines their perspective.

Challenge 5 :  This is an extension to the Challenge 2  .  Assume you are looking to build a model to predict if the text you are seeing is an issue or informational message. 
   Generate the label for each data point to see if its informational or an issue
   Build a count based matrix for each issue which reflects the number of occurrence of each word in the   text along with the label
   You can choose to restrict to a certain number of words based on your analysis. You can choose to remove the STOPWORDS from each text 
The count based matrix that you build for the entire data set should be in a format such that it can be given as an input to any of the algorithm for training and testing.

Challenge 6 :  This is an extension to the Challenge 5 . Build a TF-IDF based matrix along with the label. Explore giving that input into a logistic regression to predict if the given text is an issue. Do you see an improvement over Challenge 5 implementation?

Challenge 7 : Implement the following on the built in reuters dataset in keras
1. Tokenization 
2. Padding
3. Wordindex creation and reverse word index
4. Embedding layer for word embeddings

Challenge 8 : (NLP with Keras-TF2)
Implement the following on the built in reuters dataset in keras
1. Vectorization and Normalization
2. One Hot encoding for labels using the built in to_categorical util
3. Use these layers to build the model - Sequential, Dense, Dropout
4. Achieve a model accuracy of around 82%

Challenge 9 : (NLP with Deep Learning) - Use CNN in NLP
Here are few things we can do now.  Please try to implement them.
1. See if L1 / L2 regularizers can improve the accuracy.
2. Separate 10% data from train data and use it as validation set.  Plot the accuracy for train data and validation data against the no of epochs.
3. Use test data to predict the accuracy % of the model 

Challenge 10 - Text Sequence Generation
From a list of dino names, objective is to build a sequence model using python or tensorflow that can create new dino names based on the seed input. Please use a character level generation.

Challenge 11 - Use the yelp review data set available at https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
Create a NLP text classification model using CNNs and see if you can achieve more than 80% accuracy on TEST data.

Challenge 12 (NLP with Keras-TF2)
This is an extension to Challenge 11 to improve the results further through Hyper-parameter optimization. Use KerasClassifier (to use k-fold cross validation etc.) and RandomizedSearchCV (to find the best combination of Hyper-parameters) and fine tune the model.
