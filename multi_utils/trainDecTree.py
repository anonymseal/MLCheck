
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
import fileinput
import os
import pydot 
import re


# In[5]:


def functrainDecTree(n):

    
    df = pd.read_csv('OracleData.csv')
    data = df.values
    X = data[:, :-n]
    Y = data[:, -n:]
    model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=20, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=0)
    
    model = model.fit(X, Y)
    dump(model, 'Model/multi_label_tree.joblib')


    return model
 

