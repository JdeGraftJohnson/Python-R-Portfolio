#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 23:18:11 2022

@author: john
"""

#Import the Raw Data from JSON files

import json
import pandas as pd
import numpy as np
import pickle

n = 15000

reviews = []
with open('yelp_academic_dataset_review.json') as f:
    

    for i, line in enumerate(f):
        reviews.append(json.loads(line))
        if i+1 >= n:
            break
rev = pd.DataFrame(reviews)
rev.head()

business = []

with open('yelp_academic_dataset_business.json') as f1:
    

    for i, line in enumerate(f1):
        business.append(json.loads(line))
        if i+1 >= 200:
            break
bus = pd.DataFrame(business)
bus.head()



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





cv = CountVectorizer(max_features = n)
X = cv.fit_transform(corpus).toarray()
cv.get_feature_names()



#Convert State Names before splitting dataset into training and test set


#X = pd.DataFrame(X, columns = ['Reviews'])
#y = pd.DataFrame(y, columns = ['stars'])


y2 = rev['stars']
y2 = y2.to_numpy()
y2 = y2.astype(int)


y = []
for i, line in enumerate(y2):
        if y2[i] >= 3:
            y.append(1)
        elif y2[i] < 3:
            y.append(0)
        elif i+1 >= n:
            break
        

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


#Fit pipeline to the model
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV #Finds hyper-parameters



#kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']#A function which returns the corresponding SVC model
#def getClassifier(ktype):
 #   if ktype == 0:
        # Polynomial kernal
  #      return SVC(kernel='poly', degree=8, gamma="auto")
  #  elif ktype == 1:
 #       # Radial Basis Function kernal
#        return SVC(kernel='rbf', gamma="auto")
#    elif ktype == 2:
        # Sigmoid kernal
 #       return SVC(kernel='sigmoid', gamma="auto")
 #   elif ktype == 3:
        # Linear kernal
 #       return SVC(kernel='linear', gamma="auto")
#for i in range(4):
    # Separate data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)# Train a SVC model using different kernal
    #svclassifier = getClassifier(i) 
    #svclassifier.fit(X_train, y_train)# Make prediction
    #param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    #grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    #grid.fit(X_train,y_train)
    #y_pred = svclassifier.predict(X_test)# Evaluate our model
    #print("Evaluation:", kernels[i], "kernel")
    #print(classification_report(y_test,y_pred))
    #print(grid.best_estimator_)

from sklearn.svm import SVC
classifier = SVC(C = 10, kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

# save the model to disk
filename = 'Yelp_SVC_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
