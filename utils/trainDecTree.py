
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np
import pydot 
import re
from sklearn.tree import DecisionTreeClassifier

import fileinput
from joblib import dump, load
import graphviz

import os
import json
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pydot 
import re


def functrainDecTree():
    
    df = pd.read_csv('OracleData.csv')
    
    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]
    model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None)
    model = model.fit(X, Y)
    dump(model, 'Model/decTreeApprox.joblib')

    #tree.export_graphviz(model, out_file='tree1.dot')
    #dot_data = tree.export_graphviz(model, out_file=None, filled=True, rounded=True, special_characters=True)
    #graph = graphviz.Source(dot_data)
    #graph.render("DataBase1")

    return model


