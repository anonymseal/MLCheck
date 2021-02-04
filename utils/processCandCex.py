#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import csv as cv
from utils import Pruning
import numpy as np


# In[1]:


def funcAddCex2CandidateSet():
    df = pd.read_csv('TestDataSMT.csv')
    data = df.values
    with open('CandidateSet.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
        
        
def funcAddCexPruneCandidateSet(tree_model):
    df = pd.read_csv('TestDataSMT.csv')
    data = df.values
    
    with open('TestDataSMTMain.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
    
    df = pd.read_csv('OracleData.csv')
    #Pruning by negating the data instance
    Pruning.funcPrunInst(df, False)
    dfInst = pd.read_csv('CandidateSetInst.csv')
    dataInst = dfInst.values
    with open('CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)
    
      
    #Pruning by toggling the branch conditions
    Pruning.funcPrunBranch(df, tree_model)
    dfBranch = pd.read_csv('CandidateSetBranch.csv')
    dataBranch = dfBranch.values    
    with open('CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataBranch)  
        


# In[ ]:


def funcCheckDuplicate(pairfirst, pairsecond, testMatrix):
    pairfirstList = pairfirst.tolist()
    pairsecondList = pairsecond.tolist()
    testDataList = testMatrix.tolist()
    
    for i in range(0, len(testDataList)-1):
        if(pairfirstList == testDataList[i]):
            if(pairsecondList == testDataList[i+1]):
                return True
            #elif(pairsecondList == testDataList[i-1]):
             #   return True
    
    dfTest = pd.read_csv('TestSet.csv')
    dataTest = dfTest.values
    dataTestList = dataTest.tolist()
    for i in range(0, len(dataTestList)-1):
        if(pairfirstList == dataTestList[i]):
            if(pairsecondList == dataTestList[i+1]):
                return True
    return False 


def funcCheckCex():
    dfCandidate = pd.read_csv('CandidateSet.csv')
    dataCandidate = dfCandidate.values
    testMatrix = np.zeros((dfCandidate.shape[0], dfCandidate.shape[1]))
    
    candIndx = 0
    testIndx = 0
    
    while(candIndx < dfCandidate.shape[0]-1):
        pairfirst = dataCandidate[candIndx]
        pairsecond = dataCandidate[candIndx+1]
        #print(pairsecond)
        if(funcCheckDuplicate(pairfirst, pairsecond, testMatrix)):            
            candIndx = candIndx+2
        else:
            for k in range(0, dfCandidate.shape[1]):
                testMatrix[testIndx][k] = dataCandidate[candIndx][k]
                testMatrix[testIndx+1][k] = dataCandidate[candIndx+1][k]
            testIndx = testIndx+2    
            candIndx = candIndx+2  
    
    #print("Shape of Candidate.csv is {}".format(dfCandidate.shape[0]))
          
    with open('TestSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(testMatrix)
    
    with open('Cand-set.csv', 'w', newline='') as csvfile:
        fieldnames = dfCandidate.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(testMatrix)    
        
    #Eliminating the rows with zero values    
    dfTest = pd.read_csv('TestSet.csv')
    dfTest = dfTest[(dfTest.T != 0).any()]
    dfTest.to_csv('TestSet.csv', index = False, header = True)  
    
    #Eliminating the rows with zero values    
    dfCand = pd.read_csv('Cand-set.csv')
    dfCand = dfCand[(dfCand.T != 0).any()]
    dfCand.to_csv('Cand-set.csv', index = False, header = True)


def funcAddCexPruneCandidateSet4DNN():
    df = pd.read_csv('TestDataSMT.csv')
    data = df.values
    
    with open('TestDataSMTMain.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values  
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)
    
    df = pd.read_csv('OracleData.csv')
    #Pruning by negating the data instance
    Pruning.funcPrunInst(df, True)
    dfInst = pd.read_csv('CandidateSetInst.csv')
    dataInst = dfInst.values
    with open('CandidateSet.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(dataInst)
    
        
