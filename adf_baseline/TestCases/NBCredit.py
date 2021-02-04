

#Importing necessary files
import sys
sys.path.append('../')
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import time
from adf_baseline import symbolic_generation
from adf_data import credit
from joblib import dump, load

def func_main(sensitive_param):
    

    #Reading the dataset
    X, Y, shape, nb_classes = credit.credit_data()
    
    model = MultinomialNB()

    #Fitting the model with the dataset
    model = model.fit(X, Y)
    

    
    #Computing time
    start_time = time.time()
    #Calling the random testing approach to test strong group monotonicity
    model_job = load('adf_baseline/TestCases/NBAdult.joblib')
    fair_score = symbolic_generation.sg_main(model_job, sensitive_param)
  
    return fair_score

