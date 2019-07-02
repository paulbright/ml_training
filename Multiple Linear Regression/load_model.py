# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:09:02 2019

@author: a142400
"""

import pandas as pd
import pickle as pickle 


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#remove the first column 0
#that is take all rows of X and all columns starting from 1 
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


loaded_model = pickle.load(open("model.sav", 'rb'))

y_pred = loaded_model.predict(X_test)
print(y_pred)


