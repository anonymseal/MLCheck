

#Importing necessary files
import sys
sys.path.append('../')
import import_ipynb
import pandas as pd
import csv as cv
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
from AEQUITAS_MainFiles import Aequitas_Fully_Directed
from joblib import dump, load


def func_main(sensitive_param):
  
    model_job = load('AEQUITAS_TestCases/LogRegCredit.joblib')
    fair_score = Aequitas_Fully_Directed.aequitas_main(model_job, sensitive_param, 'LR')
  
    return fair_score

