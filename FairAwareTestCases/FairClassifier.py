
# coding: utf-8

# In[109]:


from random import seed, shuffle
import math
import os
import sys
sys.path.append('../')
from collections import defaultdict
import os,sys
#import urllib2
import numpy as np
from FairAwareTestCases import Utils as ut
from FairAwareTestCases import loss_funcs as lf
import pandas as pd
import csv as cv
#from itertools import map


def func_main():
    file_data = open('dataset.txt', 'r')
    dataset = file_data.readline()
    file_data.close()
    df = pd.read_csv(dataset)
    data = df.values

    cov = 0
    name = 'sex'
    X=[]
    Y=[]
    i=0
    sensitive = {}
    sens = []
    dfSex = df[['sex']].astype(np.int64)
    dataSex = dfSex.values
    dataSex = np.array(dataSex, dtype=int)
    dataSex = dataSex.astype(np.int64)
    dfSex.shape[0]
    train = data[:, :-1]  
    labels = data[:, -1] 


    for i in range(0, 90):
        sens.append(dataSex[i][0])
    

    for i in range(0, 90):
        temp = train[i]
        X.append(temp)    

    for i in range(0, 90):
        temp = labels[i]
        Y.append(temp)    


    
    
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype = float)
    sensitive[name] = np.array(sens, dtype = np.int64)
    loss_function = lf._logistic_loss
    sep_constraint = 0
    sensitive_attrs = [name]
    sensitive_attrs_to_cov_thresh = {name:cov}



    gamma=None

    w = ut.train_model(X, Y, sensitive, loss_function, 1, 0, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

    with open('MUTWeight.csv', 'w', newline='') as csvfile:   
        writer = cv.writer(csvfile)
        writer.writerow(df.columns.values)
        writer.writerow(w)



