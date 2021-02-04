

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
from AEQUITAS_MainFiles import Aequitas_Fully_Directed
from joblib import dump, load


def func_main(sensitive_param):
    

    #Reading the dataset
    df = pd.read_csv('Datasets/GermanCredit.csv') 

    data = df.values

    X = data[:, :-1]
    Y = data[:, -1]


    sens_feature = str(df.columns.values[sensitive_param-1])
 

    fileProt = open('protFeature.txt', 'w')
    fileProt.write(sens_feature)
    fileProt.close()
    
    model = MultinomialNB()

    #Fitting the model with the dataset
    model = model.fit(X, Y)
    

    
    #Computing time
    start_time = time.time()
    #Calling the random testing approach to test strong group monotonicity
    model_job = load('AEQUITAS_TestCases/NBCredit.joblib')
    fair_score = Aequitas_Fully_Directed.aequitas_main(model_job, sensitive_param, 'NB')
    execution_time = (time.time() - start_time)

    return fair_score

