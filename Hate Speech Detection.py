#!/usr/bin/env python
# coding: utf-8

# In[64]:


#impoting all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import CountVectorizer


# In[51]:


import re
import nltk
from nltk.util import pr
stemmer=nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words("english"))


# In[52]:


df=pd.read_csv("C:\\Users\\TUSHAR SAIN\\Downloads\\twitter_data.csv")
df.head()


# In[53]:


df["labels"]=df["class"].map({0:"Hate speech detected",1:"Offensive language detected",2:"NO hate and  offensive speech"})


# In[ ]:





# In[55]:


df=df[["tweet","labels"]]
df.head()


# In[56]:


def clean(text):
    text=str(text).lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("https?://\S+|www\.\S+",'',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub("\n","",text)
    text=re.sub("\w*\d\w*","",text)
    text=[word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["tweet"]=df["tweet"].apply(clean)
df.head()


# In[57]:


x=np.array(df["tweet"])
y=np.array(df["labels"])
cv=CountVectorizer()
x=cv.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[74]:


#logistic regression
clf=LogisticRegression()
clf.fit(x_train,y_train)


# In[68]:


#Decision_Tree_Classifier
y_pred=clf.predict(x_test)
from sklearn.metrics import accuracy_score
print("the accuracy from Logistic regression is:",accuracy_score(y_pred,y_test))


# In[73]:


clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("the accuracy  from decision tree classifier is:",accuracy_score(y_pred,y_test))


# In[77]:


#exampleand testing
data="hello"
df=cv.transform([data]).toarray()
print(clf.predict(df))


# In[ ]:




