
import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load


def functrainDecTree():
    df = pd.read_csv('OracleData.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None)
    model = model.fit(X, Y)
    dump(model, 'Model/decTreeApprox.joblib')

    return model


