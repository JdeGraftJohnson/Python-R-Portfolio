#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 22:07:34 2022

@author: john
"""

import pickle
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
import pandas as pd
import numpy as np
import pickle
import csv
from sklearn.metrics import roc_curve, auc
import seaborn as sb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Importing the dataset
df = pd.read_csv('creditcard.csv')

# Importing the dataset
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Training the Logistic Regression model on the Training set
#ROC = 0.973 , C = 10 , penalty = 'l2' found via hyperparameter
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(penalty = 'l2' , C = 10, random_state = 0)
#classifier.fit(X_train, y_train)

#Logistic Regression Grid Search define grid search - used to find the best parameters for the model
# Grid search cross validation
#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
#grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}# l1 lasso l2 ridge
#logreg=LogisticRegression()
#logreg_cv=GridSearchCV(logreg,grid,cv=10)
#logreg_cv.fit(X_train,y_train)


#print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
#print("accuracy :",logreg_cv.best_score_)

# summarize results

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Create the parameter grid based on the results of random s
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1, cv = , verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
model = rf_random.fit(X_train, y_train)


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestClassifier(n_estimators = 20, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = rf_random.best_estimator_(args, kwargs)
random_accuracy = evaluate(best_random, X_test, y_test)

# Training the K-NN model on the Training set - Results were terrible and too slow... 
#accuracy = 0.59
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making a ROC Curve due to an imbalanaced dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# generate a no skill prediction (majority class)
example_probs = [0 for _ in range(len(y_test))]  #used to create a generic ROC curve to compare classifier against
ns_auc = roc_auc_score(y_test, example_probs)
example__auc = roc_auc_score(y_test, example_probs)

clf_probs = classifier.predict_proba(X_test)
clf_probs2 = clf_probs[:, 1]
clf_auc = roc_auc_score(y_test, clf_probs2)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, example_probs)
clf_fpr, clf_tpr, _ = roc_curve(y_test, clf_probs2)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Example Classifier')
pyplot.plot(clf_fpr, clf_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()