
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import naive_bayes
import csv as cv
from sklearn.linear_model import LogisticRegression
import os
from sklearn.naive_bayes import MultinomialNB
import time
from AEQUITAS_MainFiles import Aequitas_Fully_Directed
from joblib import dump, load


#Function to calculate number of class values need to changed for promotion and demotion candidates
def funcNoOfChange(dfT, fname, j):
    countMPos = 0
    countFPos = 0
    countM = 0
    countF = 0
    
    
    for i in range(0, dfT.shape[0]):
        if(dfT.loc[dfT.index[i], fname] != 1):
            countM = countM +1
        elif(dfT.loc[dfT.index[i], fname] == 1):
            countF = countF +1
    
    for i in range(0, dfT.shape[0]):
        if((dfT.loc[dfT.index[i], fname] != 1) & (dfT.loc[dfT.index[i], 'Class'] == 1)):
            countMPos = countMPos +1
        elif((dfT.loc[dfT.index[i], fname] == 1) & (dfT.loc[dfT.index[i], 'Class'] == 1)):
            countFPos = countFPos +1
            
            
    if(j == 1):
        M = round(((countM*(countFPos)) -(countF*(countMPos)))/dfT.shape[0])
    elif(j==2):
        M = round(((countF*(countMPos)) -(countM*(countFPos)))/dfT.shape[0])
    
    
    return M 
    


# In[279]:


def funcCalDep(dfC, name):
    
    countClass1 = 0
    countClass2 = 0
    countFeatures1 = 0
    countFeatures2 = 0
    count = 0
    #tempList = []*20
    
    #Finding out the correct values for computing dependencies
    for i in range(0, dfC.shape[0]):
        if(dfC.loc[dfC.index[i], name] != 1):
            countFeatures1 = countFeatures1 +1
        elif(dfC.loc[dfC.index[i], name] == 1):
            countFeatures2 = countFeatures2 +1
            
    for i in range(0, dfC.shape[0]):
        if((dfC.loc[dfC.index[i], name] != 1) & (dfC.loc[dfC.index[i], 'Class'] == 1)):
            countClass1 = countClass1 +1
        elif((dfC.loc[dfC.index[i], name] == 1) & (dfC.loc[dfC.index[i], 'Class'] == 1)):
            countClass2 = countClass2 +1
    
    #print(countClass1)
    #print(countFeatures1)
    #Negative class is the 0 class
    
    #Computing the dependencies
    try:
        depFirst = countClass1/countFeatures1
        depSec = countClass2/ countFeatures2
        dependency = depFirst-depSec
        #print(depDiff*100)
    except ZeroDivisionError:
        print('Divide by zero')
    return dependency    
        


# In[280]:


# Function to Calculate the probabilities of belongingness to a specific class
def CalcProbabilities(fe, cl): 
        model = naive_bayes.MultinomialNB()
        y_pred = model.fit(fe, cl).predict_proba(fe)
        probs = y_pred.tolist()
        prArray = np.asarray(probs)
        return prArray


# In[295]:


def funcDataMass(dfTrain, name):
    
    flag = 0 #Indicator of dependency
    i = 0
    
    depDiff = funcCalDep(dfTrain, name)    
    if(depDiff < 0):
        flag = 1
    elif(depDiff > 0):
        flag= 2
        #Feature with 0 value has depDiff*100 less chance of getting class 1 than one with feature 1 if flag = 1
        #Feature with 1 value has depDiff*100 less chance of getting class 1 than one with feature 0 if flag = 2
    dataTrain = dfTrain.values
    X_prob = dataTrain[: , :-1]
    y_prob = dataTrain[: , -1]
    
    
    if(flag == 1):
        
         #Calculating the probabilities of belongingness to a specific class
        probsArray = CalcProbabilities(X_prob, y_prob)
        
        #Seperating Class 1 probabilities from Class 0 Probabilities    
        with open('ProbValue.csv', 'w', newline='') as csvfile:
            fieldnames = ['Prob0', 'Prob'] 
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
            writer.writerows(probsArray)
        dfNew = pd.read_csv('ProbValue.csv')
        dfNew.drop('Prob0', axis=1, inplace=True)
    
        #Merging class probabilities to the dataframe
        with open('ModFairData.csv', 'w', newline='') as csvfile:
            fieldnames = dfTrain.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
            writer.writerows(dataTrain)
        dfAg = pd.read_csv('ModFairData.csv')
        df_con = pd.concat([dfAg, dfNew], axis = 1)
        df_con.to_csv('ModFairData.csv', index= False, header= True)
        
        #Soring the dataframe based on the feature 'name' and creating  empty dataframes
        df_ModTrainProm = pd.DataFrame()
        df_ModTrainDem = pd.DataFrame()
        df_Rest1 = pd.DataFrame()
        df_Rest2 = pd.DataFrame()
        dfEnd = pd.DataFrame()
        df_ModTrain = df_con.sort_values(name)
        
        #Creating promotion candidates
        for i in range(0, df_ModTrain.shape[0]):
            if((df_ModTrain.loc[df_ModTrain.index[i], name] == 0) & (df_ModTrain.loc[df_ModTrain.index[i], 'Class'] == 0)):
                data = df_ModTrain.iloc[[i]]
                df_ModTrainProm= df_ModTrainProm.append(data)
        
        
        df_ModTrainProm = df_ModTrainProm.sort_values('Prob', ascending = False)
        #print(df_ModTrainProm)    
        
        
        #Creating demotion candidates
        for i in range(0, df_ModTrain.shape[0]):
            if((df_ModTrain.loc[df_ModTrain.index[i], name] == 1) & (df_ModTrain.loc[df_ModTrain.index[i], 'Class'] == 1)):
                data = df_ModTrain.iloc[[i]]
                df_ModTrainDem= df_ModTrainDem.append(data)
        
        
        df_ModTrainDem = df_ModTrainDem.sort_values('Prob')
        #print(df_ModTrainDem) 
        
        #Adding other data instances
        for i in range(0, df_ModTrain.shape[0]):
            if((df_ModTrain.loc[df_ModTrain.index[i], name] == 0) & (df_ModTrain.loc[df_ModTrain.index[i], 'Class'] == 1)):
                data = df_ModTrain.iloc[[i]]
                df_Rest1= df_Rest1.append(data)
        #print(df_Rest1)        
        
        for i in range(0, df_ModTrain.shape[0]):
            if((df_ModTrain.loc[df_ModTrain.index[i], name] == 1) & (df_ModTrain.loc[df_ModTrain.index[i], 'Class'] == 0)):
                data = df_ModTrain.iloc[[i]]
                df_Rest2= df_Rest2.append(data)
        #print(df_Rest2)
        #Calculate number of changes
        changes = funcNoOfChange(dfTrain, name, 1)
        #print(changes)
        
        #Changing the labels of promotion and demotion candidates
        for i in range(0, changes):
            df_ModTrainProm.loc[df_ModTrainProm.index[i], 'Class'] = 1
            df_ModTrainDem.loc[df_ModTrainDem.index[i], 'Class'] = 0
        
        dfEnd = pd.concat([df_ModTrainProm, df_ModTrainDem, df_Rest1, df_Rest2], axis = 0)
        dfEnd = dfEnd.sort_values(name)
        dfEnd.drop('Prob', axis=1, inplace=True)
        dfEnd.to_csv('DataRepository/ModTrainData.csv', index=False)
    
    elif(flag == 2):
        #Calculating the probabilities of belongingness to a specific class
        probsArray = CalcProbabilities(X_prob, y_prob)
        
        #Seperating Class 1 probabilities from Class 0 Probabilities    
        with open('ProbValue.csv', 'w', newline='') as csvfile:
            fieldnames = ['Prob0', 'Prob'] 
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
            writer.writerows(probsArray)
        dfNew = pd.read_csv('ProbValue.csv')
        dfNew.drop('Prob0', axis=1, inplace=True)
        
            
        #Merging class probabilities to the dataframe
        with open('ModFairData.csv', 'w', newline='') as csvfile:
            fieldnames = dfTrain.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
            writer.writerows(dataTrain)
        dfAg = pd.read_csv('ModFairData.csv')
        df_con = pd.concat([dfAg, dfNew], axis = 1)
        df_con.to_csv('ModFairData.csv', index= False, header= True)
        
        #Soring the dataframe based on the feature 'name' and creating  empty dataframes
        df_ModTrainProm = pd.DataFrame()
        df_ModTrainDem = pd.DataFrame()
        df_Rest1 = pd.DataFrame()
        df_Rest2 = pd.DataFrame()
        dfEnd = pd.DataFrame()
        df_ModTrain = df_con.sort_values(name)
        
        #Creating promotion candidates
        for i in range(0, df_ModTrain.shape[0]):
            if((df_ModTrain.loc[df_ModTrain.index[i], name] == 1) & (df_ModTrain.loc[df_ModTrain.index[i], 'Class'] == 0)):
                data = df_ModTrain.iloc[[i]]
                df_ModTrainProm= df_ModTrainProm.append(data)
                
        
        #print(df_ModTrainProm)
        df_ModTrainProm = df_ModTrainProm.sort_values('Prob', ascending = False)
        #print(df_ModTrainProm)    
        
        
        #Creating demotion candidates
        for i in range(0, df_ModTrain.shape[0]):
            if((df_ModTrain.loc[df_ModTrain.index[i], name] == 0) & (df_ModTrain.loc[df_ModTrain.index[i], 'Class'] == 1)):
                data = df_ModTrain.iloc[[i]]
                df_ModTrainDem= df_ModTrainDem.append(data)
        
        
        df_ModTrainDem = df_ModTrainDem.sort_values('Prob')
        #print(df_ModTrainDem) 
        
        #Adding other data instances
        for i in range(0, df_ModTrain.shape[0]):
            if((df_ModTrain.loc[df_ModTrain.index[i], name] == 1) & (df_ModTrain.loc[df_ModTrain.index[i], 'Class'] == 1)):
                data = df_ModTrain.iloc[[i]]
                df_Rest1= df_Rest1.append(data)
        #print(df_Rest1)        
        
        for i in range(0, df_ModTrain.shape[0]):
            if((df_ModTrain.loc[df_ModTrain.index[i], name] == 0) & (df_ModTrain.loc[df_ModTrain.index[i], 'Class'] == 0)):
                data = df_ModTrain.iloc[[i]]
                df_Rest2= df_Rest2.append(data)
        #print(df_Rest2)
        #Calculate number of changes
        changes = funcNoOfChange(dfTrain, name, 2)
        #print(changes)
        
        #Changing the labels of promotion and demotion candidates
        for i in range(0, changes):
            df_ModTrainProm.loc[df_ModTrainProm.index[i], 'Class'] = 1
            df_ModTrainDem.loc[df_ModTrainDem.index[i], 'Class'] = 0
        
        dfEnd = pd.concat([df_ModTrainProm, df_ModTrainDem, df_Rest1, df_Rest2], axis = 0)
        dfEnd = dfEnd.sort_values(name)
        dfEnd.drop('Prob', axis=1, inplace=True)
        dfEnd.to_csv('DataRepository/ModTrainData.csv', index=False)
        #print(dfEnd)
        
    #function to train the model
    dfAgain = pd.read_csv('DataRepository/ModTrainData.csv')
    data = dfAgain.values
    X = data[:,:-1]
    y = data[:, -1]
    model = MultinomialNB()
        
    model = model.fit(X, y)
    
    os.remove('ModFairData.csv')
    os.remove('ProbValue.csv')
    os.remove('DataRepository/ModTrainData.csv')
  
    return model




def func_main(fe):
    df = pd.read_csv('Datasets/GermanCredit.csv')
    model_job = load('AEQUITAS_TestCases/NBCredit.joblib')
    
    #Calling the random testing approach to test strong group monotonicity
    fair_score = Aequitas_Fully_Directed.aequitas_main(model_job, fe, 'DT')
    return fair_score
    



    

