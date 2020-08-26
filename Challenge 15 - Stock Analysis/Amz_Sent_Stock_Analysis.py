#*-----------------------------------------------------------------------------------------------------
#Challenge 15 (NLP with Keras-TF2)
#Go through some great case studies using NLP
#https://mobidev.biz/blog/natural-language-processing-nlp-use-cases-business
 
#Case study 7 - INTELLIGENCE GATHERING ON SPECIFIC FINANCIAL STOCKS - Here I chose Amazon
#*------------------------------------------------------------------------------------------------------
#Identify the news headlines specific to Amazon. Extract the headlines in a csv
import pandas as pd 

df1 = pd.read_csv("/content/sample_data/Amazon_news.csv", encoding='windows-1250', header=None)
df1.columns = ['Title', 'Date']
df1
#-------------------------------------------------------------------------------------------------------
#Cleaning the headlines & date columns to a format usable

import re
str=''
match=''
df1['Dateval']=df1['Date']
#df1['Dateval1']=df1['Date']

def pattern_find(in_string,in_str_pos,in_comma):
  if (in_comma<in_str_pos):
    out_start=in_comma+10
  else:
    out_start=in_comma+2
  return out_start

#Specific cleansing for the dataset - to extract the date value from the text field
for i in range(len(df1)):
  str=df1['Date'][i]
  #match = re.search(r'\w{3}\.\s\d{1,2}', str)
  #df1['Dateval1'][i]=match[0]
  cfa=0
  cfp=0
  llc=0
  cfa=str.find('CFA')
  cfp=str.find('CFP')
  llc=str.find('LLC')
  str_pos=max(cfa,cfp,llc)
  start=0
  p=str.find(',')
  if ((cfa>0) or (cfp>0) or (llc>0)):
    start=pattern_find(str,str_pos,p)
  else:
    start=p+2
  end=start+7
  df1['Dateval'][i]=str[start: end:]

df1['Dateval'] = df1.Dateval.apply(lambda text: text.replace('.',''))
df1['Dateval'] = df1.Dateval.apply(lambda text: text[0: 6:])
df1.head()

#Adding the year to the date value extracted
df1.iloc[0:99]['Dateval']=df1[0:99]['Dateval'] + ', 2020' 
df1.iloc[100:268]['Dateval']=df1[100:268]['Dateval'] + ', 2019' 
df1.iloc[268:]['Dateval']=df1[268:]['Dateval'] + ', 2018' 
df1.head()

df1.to_csv("/content/sample_data/Amazon_headlines.csv")

#Converting the date from text type to date type
from datetime import datetime

newDateList = []

for i in range(len(df1)):
  for fmt in ('%b %d, %Y', '%b %d, %Y'):
        try:
            newDate = datetime.strptime(df1['Dateval'][i], fmt).date()
            break # if format is correct, don't test any other formats
        except ValueError:
            pass
  newDateList.append(newDate) # add new date to the list

if(len(newDateList) != df1.shape[0]):
    print("Error: Rows don't match")
else:
    df1['New Date'] = newDateList # add the list to our original dataframe

df1.head()
#-------------------------------------------------------------------------------------------------------
#Sentiment analysis based on the news headlines 
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

results = []

for headline in df1['Title']:
    pol_score = SIA().polarity_scores(headline) # run analysis
    pol_score['headline'] = headline # add headlines for viewing
    results.append(pol_score)

#results

#Score being appended to the original DataFrame
df1['Score'] = pd.DataFrame(results)['compound']

#Creates a daily score by aggregatin the scores of the individual articles in each day
df2 = df1.groupby(['New Date']).sum() 

#-------------------------------------------------------------------------------------------------------
#Inorder to understand the impact of news on the stock price, historical stock values are downloaded
#(for the same period we have news headlines updates)

#Reading the historical stock price for Amazon
dfEodPrice = pd.read_csv("/content/sample_data/AMZN.csv")

#Converting the Date column to datetime type
dfEodPrice['Date'] = dfEodPrice['Date'].astype('datetime64[ns]')

#Dropping unwanted rows
dfEodPrice2 = dfEodPrice.drop(['Open', 'High','Low','Close','Volume'], axis=1) 
dfEodPrice2.set_index('Date', inplace=True) #Setting Date coloumn as index

#Calculating returns - Divide today’s prices by yesterday’s
dfEodPrice2['Returns'] = dfEodPrice2['Adj Close']/dfEodPrice2['Adj Close'].shift(1) - 1 

#One-day lagged sentiment score - allows to compare today’s article headlines to tomorrow’s stock returns
df2['Score(1)'] = df2.shift(1)

#Match the daily returns with the lagged sentiment score based on date (index column is date)
dfReturnsScore = pd.merge(dfEodPrice2[['Returns']], df2[['Score(1)']], left_index=True, right_index=True, how='left')
dfReturnsScore.fillna(0, inplace=True) # replace NaN with 0 permanently

#-------------------------------------------------------------------------------------------------------
#Plot the score against the returns
dfReturnsScore.plot(x="Score(1)", y="Returns", style="o")

#Assume that sentiment score of > 0.5 or < -0.5 has a predictive value on only tomorrow’s daily returns
dfReturnsScore2 = dfReturnsScore[(dfReturnsScore['Score(1)'] > 0.5) | (dfReturnsScore['Score(1)'] < -0.5)]

#Plot the score against the returns with the data again							 
dfReturnsScore2.plot(x="Score(1)", y="Returns", style="o")

#Checking co-relation
dfReturnsScore2['Returns'].corr(dfReturnsScore2['Score(1)'])

#Correlation coefficient is -0.042. That’s close to 0. This means article headlines alone do not have 
#any predictive value for tomorrow’s stock returns.

#-------------------------------------------------------------------------------------------------------
