

#Importing necessary files
import sys
sys.path.append('../')
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time
from adf_baseline import symbolic_generation
from adf_data import census
from joblib import dump, load


def func_main(sensitive_param):
    

    #Reading the dataset
    X, Y, shape, nb_classes = census.census_data()

    model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       random_state=None, splitter='best')

    #Fitting the model with the dataset
    model = model.fit(X, Y)
    

    
    #Computing time
    start_time = time.time()
    #Calling the random testing approach to test strong group monotonicity
    model_job = load('adf_baseline/TestCases/DecTreeAdult.joblib')
    fair_score = symbolic_generation.sg_main(model_job, sensitive_param)

    return fair_score

