Challenge 1: Tokenization : You have a collection of tickets enclosed below. Apply tokenization using python nltk and generate a vocabulary of words in the ticket. 

Challenge 2 :  Tokenization, FDIST, ngrams This is an extension to the Challenge 1  .  Once you have tokenized, find the n-grams using FDIST function. FDIST and ngrams will give you the laundry list but has a lot of noise, hence the objective is to extract the key frequently occurring ngrams (where n>1 and n<=4) based on n=1 most frequently occurring terms. Assume that you need to be able to present to the Customer key issues found in the ticket database. Please feel free to explore the inclusion of STOPWORDS in this solution

Challenge 3 :  Assume the Customer has a collection of requirements that have been implemented in the past.  Customer is looking for a mechanism to identify if the new incoming requirement is a match to one of the implemented ones. The challenge is Customers organization spends quite a bit of time understanding if they have done it before. In the bigger picture Customer wants to leverage the learning of the past before implementing a similar requirement. Enclosed is a sample set of requirements, come up with an approach to identify the match against the incoming requirement, you need to select an appropriate measure to reflect the match percentage. You must include STOPWORDS in this solution.

Challenge 4 :  You have access to a html file called leaderTalk.html which has leaders talk about different dimensions both personal and official, including why they think Cognizant is their favorite organization. The objective is to parse the content related to why they think Cognizant is their favorite organization and generate word cloud that outlines their perspective.

Challenge 5 :  This is an extension to the Challenge 2  .  Assume you are looking to build a model to predict if the text you are seeing is an issue or informational message. For e.g. job completed is an informational message whereas job abended is an issue.  The objective is to predict the issue.  
The task is multi fold
   Generate the label for each data point to see if its informational or an issue
   Build a count based matrix for each issue which reflects the number of occurrence of each word in the text along with the label
   You can choose to restrict to a certain number of words based on your analysis
   You can choose to remove the STOPWORDS from each text 
   The count based matrix that you build for the entire data set should be in a format such that it can be given as an input to any of the algorithm for training and testing.
