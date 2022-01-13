# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import scipy as scp
import numpy as np
import matplotlib.pyplot as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

"""Wrapper for using the Scikit-Learn API with Keras models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import types

import numpy as np

from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.generic_utils import has_arg
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.util.tf_export import keras_export

#Importing the dataset
training_set = pd.read_csv('train.csv')
# =============================================================================
# x_train = training_set.iloc[:, training_set.columns != 'label'].values
# y_train = training_set.iloc[:, 0].values
# =============================================================================

x = training_set.iloc[:, training_set.columns != 'label'].values
y = training_set.iloc[:, 0].values


# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


x_train = x_train.astype("float32") / 255


# =============================================================================
# test_set = pd.read_csv('test.csv')
# x_test = test_set.iloc[:, :]
# =============================================================================
x_test = x_test.astype("float32") / 255

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []




# Merge inputs and targets
inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = pd.DataFrame(sc.fit_transform(x_train))
# =============================================================================

  
# =============================================================================
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(42000, 784)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])
# 
# =============================================================================
# =============================================================================
# print(y_train.shape)
# y_cat = to_categorical(y_train)
# =============================================================================

# =============================================================================
# model = Sequential()
# model.add(Dense(60, input_shape = (784,), activation = "relu"))
# model.add(Dense(15, activation = "relu"))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation = "softmax"))
# model.compile(Adam(lr = 0.01), "categorical_crossentropy", metrics = ["accuracy"])
# model.summary()
# =============================================================================

# =============================================================================
# model.fit(x_train, y_cat, verbose=1, epochs=10)
# =============================================================================
# Predicting a new result

num_folds = 10
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True,random_state=42)

for train, test in kfold.split(inputs, targets):
  # create model
	model = Sequential()
	model.add(Dense(12, input_dim=784, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(inputs[train], targets[train], epochs=10, batch_size=128, verbose=0)
	# evaluate the model
	scores = model.evaluate(inputs[test], targets[test], verbose=0)
	#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	#cvscores.append(scores[1] * 100)
#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
accuracy = cross_val_score(model, inputs, targets, scoring='accuracy', cv = 10) 
print(accuracy)
#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)



# K-fold Cross Validation model evaluation

# =============================================================================
# fold_no = 1
# for train, test in kfold.split(inputs,targets):
#   fold_no = fold_no + 1
#   # Define the model architecture
#   model = Sequential()
#   model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(42000,784)))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   model.add(Flatten())
#   model.add(Dense(256, activation='relu'))
#   model.add(Dense(128, activation='relu'))
#   model.add(Dense(9, activation='softmax'))
#   
#   fold_no = fold_no + 1
#   # Compile the model
#   model.compile(loss=loss_function,
#                 optimizer=optimizer,
#                 metrics=['accuracy'])
# 
# 
#   # Generate a print
#   print('------------------------------------------------------------------------')
#   print(f'Training for fold {fold_no} ...')
# 
#   # Fit data to model
#   history = model.fit(inputs[train], targets[train],
#               batch_size=batch_size,
#               epochs=no_epochs,
#               verbose=verbosity)
# 
#   # Generate generalization metrics
#   scores = model.evaluate(inputs[test], targets[test], verbose=0)
#   print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
#   acc_per_fold.append(scores[1] * 100)
#   loss_per_fold.append(scores[0])
# 
# =============================================================================
  # Increase fold number
  
class BaseWrapper(object):
    def __init__(self, build_fn=None, **sk_params):
    self.build_fn = build_fn
    self.sk_params = sk_params
    self.check_params(sk_params)

  def check_params(self, params):
    """Checks for user typos in `params`.
    Arguments:
        params: dictionary; the parameters to be checked
    Raises:
        ValueError: if any member of `params` is not a valid argument.
    """
    legal_params_fns = [
        Sequential.fit, Sequential.predict, Sequential.predict_classes,
        Sequential.evaluate
    ]
    if self.build_fn is None:
      legal_params_fns.append(self.__call__)
    elif (not isinstance(self.build_fn, types.FunctionType) and
          not isinstance(self.build_fn, types.MethodType)):
      legal_params_fns.append(self.build_fn.__call__)
    else:
      legal_params_fns.append(self.build_fn)

    for params_name in params:
      for fn in legal_params_fns:
        if has_arg(fn, params_name):
          break
      else:
        if params_name != 'nb_epoch':
          raise ValueError('{} is not a legal parameter'.format(params_name))