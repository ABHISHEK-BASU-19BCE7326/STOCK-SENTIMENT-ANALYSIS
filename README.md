# STOCK-SENTIMENT-ANALYSIS

## INDEX
1. INTRODUCTION
2. AIM
3. WORKING WITH DATA SET
4. EXTRACTING DATA
5. DATA CLEANING
6. PREDICTION/ANALYSIS USING ML
7. RESULTS
8. CONCLUSION


## Key words:
1. DJIA- Dow Jones Industrial Average
2. Open position: holding a stock or longing/shorting a stock
3. Close position: Closing the open position


- INTRODUCTION:

In the stock market where a stock doesn't necessarily move in accordance to it’s innate value in terms of the fundamentals but is more so price driven depending on the market maker’s interest, something as fragile as a small news might change the direction of movement of a stock.

In the dataset obtained from https://www.kaggle.com/aaron7sun/stocknews ,the “news data”, that is, the headlines, they have been obtained from https://www.reddit.com/r/worldnews?hl and the stock data, to see whether the par (labelled with a 1 or 0) has been obtained from the “Dow Jones Industrial Average(DJIA)”.

The analysis of the data gives us a comprehensive view and understanding about what kind of news really affected.

The Dow Jones Industrial Average, Dow Jones, or simply the Dow, is a price-weighted measurement stock market index of the 30 prominent companies listed on stock exchanges in the United States.

Analysis of the stock headlines gives us a view of what kind of news headlines really affected in the market rallying down for those 30 stocks on an overall, thus letting us make informed decisions if one wants to put the understanding to practical use and do day-trading, as news is most influential for day-trading.

Using NLP we are able to run a few classification algorithms on the dataset we have and produce predictions in binary to see whether the price rallies up(giving a predicted value of 1) or if the price has decreased(giving a predicted value of 0).



- AIM:

The aim of this project is to gain a compendious overview of how daily news influences how a stock or a collection of similar stocks(blue chip stocks in this case, as we are taking the Dow Jones Average into consideration) perform or change their direction of movement, be it positive or negative in response to news.
This can be crucial information to gain an understanding about, as in the long run, it might help a trader make informed decisions, whether to open a position and make profits or close his existing position to cut down on losses.

Even though the dataset obtained is a little old(2016-2008), it is still crucial information for us to train and build models on.


- WORKING WITH DATASET:

The dataset is a labelled dataset, meaning we know what kind of news in the past has led to an increase or decrease of the closing average for the DJIA.
The dates range from 2000-01-03(YYYY/MM/DD) to 2017-07-01.

There are a minimum 2 gap days from week to week, which is the weekends, that being Saturday and Sunday.
Other days like national holidays could have set in for a few months, for which the market could have been on a holiday.

We have 25 headlines for each day relative to which for the closing price of the DJIA, we have a label 1 or 0.




- EXTRACTING DATA:

Every single row in the dataset is essential to us, as any and all news we can obtain to help establish a relation and make predictions can be useful.

There are 27 columns in total, of which 25 are the headlines for each day and the other 2 rows are the dates and the marker(label) for whether the DJIA increased/stayed the same(1) or decreased.


- DATA CLEANING:

The punctuations in the dataset for each of the headlines are removed and converted to lowercase as they add no value to the dataset but rather add complexity to the prediction because the bag of words created using CountVectorizer shall judge the a word with different punctuation, say one starting with a capital letter and the other starting with a small letter as two different words.


- PREDICTION/ANALYSIS USING ML:

All the rows in the dataset are joined to be able to vectorize the words, so that the prediction is done on the entire dataset in a unified manner.

Then, CountVectorizer is used to convert all the sentences into vectors.
```
CountVectorizer is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs 
in the entire text.
```

Once that is done, we implement the Bag of Words created using the CountVectorizer.

The 4 techniques used to judge the relative accuracy of prediction for this project are Random Forest Classifier, Multinomial Naive Bayes, Decision Tree and Support Vector Machine.

The algorithms were found to give the best accuracy for such labeled dataset where huge amounts of text based data was present, which is why we’ve tried to perform on and compare the accuracies of these four algorithms.
```
Random forest is a flexible, easy to use machine learning algorithm that produces, even without hyper-parameter tuning, a great
result most of the time.It is also one of the most used algorithms, because of its simplicity and diversity
```
```
Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP). 
The algorithmis based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article.It
calculates the probability of each tag for a given sample and then gives the tag with the highest probability as output.
```
```
Decision Tree algorithm belongs to the family of supervised learning algorithms.Unlike other supervised learning algorithms,
the decision tree algorithm can be used for solving regression and classification problems too.The goal of using a Decision Tree 
is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules
inferred from prior data(training data).
```
```
SVM(Support Vector Machine) offers very high accuracy compared to other classifiers such as logistic regression, and decision 
trees.It is known for its kernel trick to handle nonlinear input spaces.It is used in a variety of applications such as face 
detection, intrusion detection, classification of emails, news articles and web pages, classification of genes, and handwriting 
recognition.
```


- RESULTS:

While during multiple iterations the 4 algorithms averaged around the same accuracy of about 85.4%, at times Multinomial Naive Bayes and Random Forest Classifier showed a difference.

Both the algorithms have a close match with the prediction accuracies closing in at 84.6% and 84.3% for Random Forest Classifier and Multinomial Naive Bayes respectively.

### RandomForestClassifier: 

:: 84.3%


|             |precision |recall   |f1 score  |support  |
|-------------|----------|---------|----------|---------|
|0            |0.94      |0.73     |0.82      |186      |
|1            |0.79      |0.96     |0.86      |192      |
|accuracy     |          |         |0.85      |378      |
|macro avg    |0.87      |0.84     |0.84      |378      |
|weighted avg |0.86      |0.85     |0.84      |378      |





### MultinomialNaiveBayes:

::84.6%


|             |precision |recall   |f1 score  |support  |
|-------------|----------|---------|----------|---------|
|0            |0.94      |0.73     |0.82      |186      |
|1            |0.79      |0.95     |0.86      |192      |
|accuracy     |          |         |0.84      |378      |
|macro avg    |0.86      |0.84     |0.84      |378      |
|weighted avg |0.86      |0.85     |0.84      |378      |

### Decision Tree:

::85.4%


|             |precision |recall   |f1 score  |support  |
|-------------|----------|---------|----------|---------|
|0            |0.80      |0.79     |0.79      |186      |
|1            |0.80      |0.81     |0.80      |192      |
|accuracy     |          |         |0.80      |378      |
|macro avg    |0.80      |0.80     |0.80      |378      |
|weighted avg |0.80      |0.80     |0.80      |378      |

### Support Vector Machine:

::85.4%

|             |precision |recall   |f1 score  |support  |
|-------------|----------|---------|----------|---------|
|0            |0.83      |0.85     |0.84      |186      |
|1            |0.85      |0.83     |0.84      |192      |
|accuracy     |          |         |0.84      |378      |
|macro avg    |0.84      |0.84     |0.84      |378      |
|weighted avg |0.84      |0.84     |0.84      |378      |

- CONCLUSION:

From the binary numpy array output that we were able to receive, in accordance with the headline which contrasted with the output label, we can infer what kind of news results in what kind of output for a stock.

Taking the probability of trades into consideration, since the accuracy levels are over 80%, while such an analysis can be relied on as an indicator, it should not be used as the sole ingredient to make trades.

## CODE
#!/usr/bin/env python
# coding: utf-8

# In[20]:


###https://www.kaggle.com/aaron7sun/stocknews
```
import pandas as pd
```

# In[21]:

```
df=pd.read_csv('C:/Users/akhil/Documents/DSProject/Data.csv', encoding ='ISO-8859-1')
```

# In[22]:


#train and test
```
train=df[df['Date']<'20150101']
test=df[df['Date']>'20141231']
```

# In[23]:


#Removing punctuations
```
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
```
#Renaming column names for ease of access
```
list1=[i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns=new_Index
data.head(5)
```

# In[24]:


#Converting headlines to lower case
```
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)
```

# In[25]:


#joining all the sentences in a row, with all the special character removed earlier, to be able to vectroize
```
' '.join(str(x) for x in data.iloc[0,0:25])
```

# In[26]:


#iterting through all the entries and joining them, appending and storing them in a list
```
headlines=[]

for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
```

# In[27]:

```
headlines
```

# In[28]:


##CountVectorizer-responsible for converting all sentences into vectors
```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble  import RandomForestClassifier
```

# In[29]:


##implementing Bag of words
```
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)
```

# # Random Forest Classifier

# In[30]:


##Implement RandomForest Classifier
```
randomclassifier=RandomForestClassifier(n_estimators=200, criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])
```

# In[31]:


##Prediction for the Test Dataset
```
test_transform=[]
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=countvector.transform(test_transform)    
predictions=randomclassifier.predict(test_dataset)
```

# In[32]:

```
predictions
```

# In[33]:


#importing library to check accuracy
```
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

# In[34]:

```
matrix=confusion_matrix(test['Label'], predictions)
print(matrix)
score=accuracy_score(test['Label'], predictions)
print(score)
report=classification_report(test['Label'], predictions)
print(report)
```

# # Multinomial Naive Bayes

# In[35]:

```
from sklearn.naive_bayes import MultinomialNB
```

# In[36]:

```
naive=MultinomialNB()
naive.fit(traindataset, train['Label'])
```

# In[37]:


##Prediction for the Test Dataset
```
test_transform=[]
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=countvector.transform(test_transform)    
predictionsnb=naive.predict(test_dataset)
```

# In[38]:

```
predictionsnb
```

# In[39]:

```
matrix=confusion_matrix(test['Label'], predictionsnb)
print(matrix)
score=accuracy_score(test['Label'], predictions)
print(score)
report=classification_report(test['Label'], predictionsnb)
print(report)
```

# # Decision Tree

# In[40]:

```
from sklearn.tree import DecisionTreeClassifier
decision=DecisionTreeClassifier()
decision.fit(traindataset,train['Label'])
```

# In[41]:


##Prediction for the Test Dataset
```
test_transform=[]
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=countvector.transform(test_transform)    
predictionstree=decision.predict(test_dataset)
```

# In[42]:

```
predictionstree
```

# In[43]:

```
matrix=confusion_matrix(test['Label'], predictionstree)
print(matrix)
score=accuracy_score(test['Label'], predictions)
print(score)
report=classification_report(test['Label'], predictionstree)
print(report)
```

# # Support Vector Machine

# In[44]:

```
from sklearn import svm
supportvec = svm.SVC(kernel='linear')
supportvec.fit(traindataset,train['Label'])
```

# In[45]:


##Prediction for the Test Dataset
```
test_transform=[]
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=countvector.transform(test_transform)    
predictionssvm=supportvec.predict(test_dataset)
```

# In[46]:

```
predictionssvm
```

# In[47]:

```
matrix=confusion_matrix(test['Label'], predictionssvm)
print(matrix)
score=accuracy_score(test['Label'], predictions)
print(score)
report=classification_report(test['Label'], predictionssvm)
print(report)
```

# In[ ]:
