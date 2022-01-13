#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 22:07:34 2022

@author: john
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

import json
import pandas as pd
import numpy as np

import pickle


start = 15000
end = 30000 
reviews = []

with open('yelp_academic_dataset_review.json') as f:
    
    
    for i, line in enumerate(f):
            if i > start and i < end:
                reviews.append(json.loads(line))
            elif i > end:
                break
rev = pd.DataFrame(reviews)
rev.head()

#Question 1 - Can you predict what rating a customer would give a place based on their review and useful, funny and cool columns?
# Model - Regression 

#Select what data to work with
#Reviews, Date posted, Business ID


# Importing the dataset
#y = rev.pop('stars')
X = rev.iloc[:, 7].values


#Convert Reviews Column using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.stem.porter import PorterStemmer


corpus = []
ps = PorterStemmer()
for words in X:
    num = ps.stem(words)
    corpus.append(num)

cv = CountVectorizer(max_features = start)
X = cv.fit_transform(corpus).toarray()
cv.get_feature_names()

#Convert State Names before splitting dataset into training and test set

#X = pd.DataFrame(X, columns = ['Reviews'])
#y = pd.DataFrame(y, columns = ['stars'])

y2 = rev['stars']
y2 = y2.to_numpy()
y2 = y2.astype(int)


y = []
for j, line in enumerate(y2):
        if y2[j] >= 3:
            y.append(1)
        elif y2[j] < 3:
            y.append(0)
        elif i+1 >= start:
            break
        

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 42)

loaded_model = pickle.load(open('Yelp_SVC_model.sav', 'rb'))
result = loaded_model.score(X, y)
print(result)