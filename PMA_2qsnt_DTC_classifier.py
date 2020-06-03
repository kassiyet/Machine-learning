#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys
import time
import json


# In[2]:


# read the file
df = pd.read_csv('PMA_blockbuster_movies.csv')
df.head()


# In[3]:


# droping the features which seems no influence on prediction
""" 'poster_url' is not influensing to prediction, unless people are not captured by adds and word of mouth
    
    'genres','Genre_1','Genre_2','Genre_3' are worthwhile features for prediction, however, I dont know how to transform 
     them into the numbers. through 'dummies' it is going to be many features and since the same genre occur in 
     'Genre_1','Genre_2','Genre_3' it seem to be repeatitive. 
     
     'release_date','year' - is also not useful for prediction, since doesn't make sence.
     
     'title' - at some point make sense, but not much.
     
     'worldwide_gross'- since it is one of the target variable and it is similar to 'Adjusted' column it shouldnt be
      in the set of X variables.
"""

df = df.drop(columns=['studio','poster_url','genres','Genre_1','Genre_2',
                      'Genre_3','release_date','title','worldwide_gross','year'])


# In[4]:


# to split the values of Rating by types

df = pd.get_dummies(df, columns=["rating"], prefix=["rating_type"])


# In[5]:


df.head()


# In[6]:


# drop the % character from cloumn '2015_inflation'
# drop the $ character from column 'adjusted'

df['2015_inflation'] = df['2015_inflation'].str[:-1].astype(float)
df['adjusted'] = df['adjusted'].str[1:].astype(str)


# In[7]:


df.head()


# In[8]:


# drop ',' from column 'adjusted' and change the type of column to Float
df['adjusted'] = df['adjusted'].str.replace('\,','')
df['adjusted'] = df['adjusted'].astype(float)


# In[9]:


df.head()


# In[10]:


#for classification model the column values are replaced by 1 if it is more than mean of column, otherwise to 0
df['adjusted'] = np.where(df['adjusted'] > df["adjusted"].mean(), 1, 0)


# In[11]:


df


# In[12]:


df = df.dropna()


# In[13]:


# splitting data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df.loc[:, df.columns != 'adjusted'], df['adjusted'], 
		test_size = 0.2, random_state=5)  # X is “1:” and Y is “[0]”

# print the shapes to check everything is OK
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[14]:


from sklearn.tree import DecisionTreeClassifier as DTC

# a decision tree model with default values
dtc = DTC()

# fit the model using some training data
dtc_fit = dtc.fit(X_train, Y_train)

# generate a mean accuracy score for the predicted data
train_score = dtc.score(X_train, Y_train)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))


#################################################################
#####                                                       #####
#####  TESTING PART - ONLY RUN WHEN THE MODEL IS TUNED!!    #####
#####                                                       #####
#################################################################

# predict the test data
predicted = dtc.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = dtc.score(X_test, Y_test)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(test_score, 4)))


# In[15]:


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
get_ipython().run_line_magic('matplotlib', 'inline')

np.set_printoptions(precision=2)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalise=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          multi=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalise=True`.
    """
    if not title:
        if normalise:
            title = 'Normalised confusion matrix'
        else:
            title = 'Confusion matrix, without normalisation'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    if multi==True:
    	classes = classes[unique_labels(y_true, y_pred)]
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor");

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax

# predict the test data - change model for whatever name you are using for the model
predicted = dtc.predict(X_train)

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_train, predicted, classes=["0", "1"])

# Plot normalised confusion matrix
plot_confusion_matrix(Y_train, predicted, classes=["0", "1"], normalise=True)


# In[16]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC

tuned_parameters = [{'criterion': ['gini', 'entropy'],
                     'max_depth': [3, 5, 7],
                     'min_samples_split': [3, 5, 7],
                     'max_features': ["sqrt", "log2", None]}]

scores = ['accuracy', 'f1_macro']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(DTC(), tuned_parameters, cv=5,
                       scoring= score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[26]:


from sklearn.tree import DecisionTreeClassifier as DTC

# a decision tree model with default values
dtc = DTC(criterion= 'gini', max_depth= 7, max_features= 'log2', min_samples_split = 7)

# fit the model using some training data, learn the rules 
dtc_fit = dtc.fit(X_train, Y_train)

# generate a mean accuracy score for the predicted data
train_score = dtc.score(X_train, Y_train)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))


# In[27]:



# predict the test data
predicted = dtc.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = dtc.score(X_test, Y_test)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(test_score, 4)))


# In[28]:


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
get_ipython().run_line_magic('matplotlib', 'inline')

np.set_printoptions(precision=2)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalise=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          multi=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalise=True`.
    """
    if not title:
        if normalise:
            title = 'Normalised confusion matrix'
        else:
            title = 'Confusion matrix, without normalisation'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    if multi==True:
    	classes = classes[unique_labels(y_true, y_pred)]
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor");

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax

# predict the test data - change model for whatever name you are using for the model
predicted = dtc.predict(X_test)

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])

# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], normalise=True)

