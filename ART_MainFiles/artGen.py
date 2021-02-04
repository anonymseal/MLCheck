
# coding: utf-8

# In[ ]:




#Importing necessary files
import pandas as pd
import numpy as np
import random as rd
import csv as cv
import os
import math
import scipy.stats as st
from scipy.spatial import distance
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




#Function to check whether a generated test pair already exists in the test suite
def chkPairBel(tempMatrix, noAttr, n):
    
    firstTest = np.zeros((noAttr, ))
    secTest = np.zeros((noAttr, ))

    if(n == 0):
        dfT = pd.read_csv('TestDataSet.csv')
    else:
        dfT = pd.read_csv('CandTestDataSet.csv')
    tstMatrix = dfT.values
    
    for i in range(0, noAttr):
        firstTest[i] = tempMatrix[0][i]

    firstTestList = firstTest.tolist()
    secTestList = secTest.tolist()
    testMatrixList = tstMatrix.tolist()
    for i in range(0, len(testMatrixList)-1):
        if(firstTestList == testMatrixList[i]):
            if(secTestList == testMatrixList[i+1]):
                return True
    return False  


def chkAttack(model, target_class):
    
    cexPair = ()
    
    dfTest = pd.read_csv('TestDataSet.csv')
    dataTest = dfTest.values
    i = 0
    X = torch.tensor(dataTest, dtype=torch.float32)
    while(i < dfTest.shape[0]-1):
        #for j in range(0, dfTest.shape[1]):
            #firstTest[0][j] = dataTest[i][j]
        predict_prob = model(X[i].view(-1, X.shape[1]))
        pred_class = int(torch.argmax(predict_prob))
        if(pred_class != 4 and pred_class != 5):
            cexPair = (X[i])
            return cexPair
        i = i+1
    return cexPair


def funcDetFur():
    dfTest = pd.read_csv('CandTestDataSet.csv')
    dfCand = pd.read_csv('TestDataSet.csv')
    
    cand_furthest = np.zeros((1, dfTest.shape[1]))

    maxDist = 0
    i = 0
    j = 0
    while(i < dfCand.shape[0]):
        x1 = dfCand.iloc[i]
        while(j < dfTest.shape[0]):
            x2 = dfTest.iloc[j]
            dist = distance.euclidean(x1, x2)
            if(dist >= maxDist):
                for ind in range(0, dfTest.shape[1]):
                    cand_furthest[0][ind] = x1[ind]
                maxDist = dist    
            j = j+1
        i = i+1
        
    return cand_furthest
    
    


def funcGenInstPair(df, tempMatrix, min_feature_val, max_feature_val, trigger):
    
    
    noOfAttr = df.shape[1]-1
    
    #Generating the first test instance (x) of the pair, refer to line 3 of ranTest algo
    if(trigger == 'T1'):
        start = 2
        tempMatrix[0][0] = 673.723
        tempMatrix[0][1] = 43.693
    elif(trigger == 'T2'):
        start = 3
        tempMatrix[0][0] = 673.723
        tempMatrix[0][1] = 43.693
        tempMatrix[0][2] = 83.484
    elif(trigger == 'T3'):
        start = 5
        tempMatrix[0][0] = 673.723
        tempMatrix[0][1] = 43.693
        tempMatrix[0][2] = 83.484
        tempMatrix[0][3] = 2137.505
        tempMatrix[0][4] = 2137.505
    elif(trigger == 'T4'):
        start = 7
        tempMatrix[0][0] = 673.723
        tempMatrix[0][1] = 43.693
        tempMatrix[0][2] = 83.484
        tempMatrix[0][3] = 2137.505
        tempMatrix[0][4] = 2137.505  
        tempMatrix[0][5] = 43.693
        tempMatrix[0][6] = 83.484   

    
    for i in range(start, noOfAttr):
            
        fe_type = df.dtypes[i]
        fe_type = str(fe_type)
            
        if('int' in fe_type):
            tempMatrix[0][i] = rd.randint(min_feature_val[i], max_feature_val[i])
        else:
            tempMatrix[0][i] = rd.uniform(min_feature_val[i], max_feature_val[i])
            
     
    return tempMatrix
    
    
    
def funcGenTestStr(df, model, MAX_SAMPLES, trigger, target_class):
    
    
    INI_SAMPLES = 100
    POOL_SIZE = 20
    fe_type = ''
    test_count = 0
    cand_count = 0
    
    #counting samples
    count = 0 
    #counting number of non monotonic instances
    nmon_count = 0
    
    
    #Initializing the test set and the arrays which will hold the minimum and maximum values of each feature
    noOfAttr = df.shape[1]-1
    tempMatrix = np.zeros((1, noOfAttr))
    tempMatrixCand = np.zeros((1, noOfAttr))
    
    
    min_feature_val = np.zeros((noOfAttr, ))
    max_feature_val = np.zeros((noOfAttr, ))
    
    #Getting the maximum and minimum feature values for each features which is used to generate valid test data
    for i in range(0, noOfAttr):
        min_feature_val[i] = df.iloc[:, i].min()
        max_feature_val[i] = df.iloc[:, i].max()
    
    
    
    #Test data schema preparing
    with open('TestDataSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
    
    #Candidate data schema preparing
    with open('CandTestDataSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)


    #Defining a new column which will indicate which data instance belongs to which pair
    dfAg = pd.read_csv('TestDataSet.csv')
    dfAg.drop('Class', axis=1, inplace=True)
    dfAg.to_csv('TestDataSet.csv', index= False, header=True)
    
    #Defining a new column which will indicate which data instance belongs to which pair
    dfAg = pd.read_csv('CandTestDataSet.csv')
    dfAg.drop('Class', axis=1, inplace=True)
    dfAg.to_csv('CandTestDataSet.csv', index= False, header=True)
    

    
    #Refer to line 2 of ranTest algo
    while(count < INI_SAMPLES):
        #tempMatrix will hold x and x'
        
        tempMatrix = funcGenInstPair(df, tempMatrix, min_feature_val, max_feature_val, trigger)
        #Adding the test pair into the test suite, if the pair does not belong to the test suite     
        if(chkPairBel(tempMatrix, noOfAttr, 0) == False):
            
            with open('TestDataSet.csv', 'a', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerows(tempMatrix) 
                
            count = count+1
            
    while(test_count < MAX_SAMPLES):
        
        while(cand_count < POOL_SIZE):
            
            tempMatrixCand = funcGenInstPair(df, tempMatrixCand, min_feature_val, max_feature_val, trigger)
            
            if(chkPairBel(tempMatrixCand, noOfAttr, 1) == False):
            
                with open('CandTestDataSet.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(tempMatrixCand)
                    
                cand_count = cand_count+1        
        
        cand_furthest = funcDetFur()
        
        with open('TestDataSet.csv', 'a', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerows(cand_furthest)
            
        test_count = test_count+1    
    
            
    #Checking the monotonicity of the generated test  cases
    cexPair= chkAttack(model, target_class)  
       
    return cexPair

    


def funcMain(MAX_SAMPLES, model, df, arch, target_class, trigger):


    cexPair = funcGenTestStr(df, model, MAX_SAMPLES, trigger, target_class)    
    return cexPair

