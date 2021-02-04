

#Importing necessary files
import sys
sys.path.append('../')
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
from adf_baseline import symbolic_generation
from adf_data import credit
from joblib import dump, load

def func_main(sensitive_param):
    

    #Reading the dataset
    X, Y, shape, nb_classes = credit.credit_data()
    
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                           class_weight=None, random_state=10, solver='lbfgs', max_iter=5000, multi_class='auto', 
                           verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

    #Fitting the model with the dataset
    model = model.fit(X, Y)
    

    
    #Computing time
    start_time = time.time()
    #Calling the random testing approach to test strong group monotonicity
    model_job = load('adf_baseline/TestCases/LogRegCredit.joblib')
    fair_score = symbolic_generation.sg_main(model_job, sensitive_param)

    return fair_score

