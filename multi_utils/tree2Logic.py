
# coding: utf-8

# In[ ]:



# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np

from sklearn.tree import DecisionTreeClassifier

import fileinput

import graphviz

import os
import json
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import pydot 
import re


# In[2]:


from sklearn.tree import _tree
def tree_to_code(tree, feature_names):
    
    f = open('TreeOutput.txt', 'w')
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    f.write("def tree({}):".format(", ".join(feature_names)))
    f.write("\n")
    
    pred_arr = np.zeros((tree_.n_outputs))
    
    def recurse(node, depth):
        indent = "  " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            threshold = round(threshold,5)
            
            f.write("{}if {} <= {}:".format(indent, name, threshold))
            f.write("\n")
            
            f.write("{}".format(indent)+"{")
            f.write("\n")
            
            recurse(tree_.children_left[node], depth + 1)
            
            f.write("{}".format(indent)+"}")
            f.write("\n")
            
            f.write("{}else:  # if {} > {}".format(indent, name, threshold))
            f.write("\n")
            
            f.write("{}".format(indent)+"{")
            f.write("\n")
            
            recurse(tree_.children_right[node], depth + 1)
            
            f.write("{}".format(indent)+"}")
            f.write("\n")
            
        else:
            for i in range(0, tree_.n_outputs):
                pred_arr[i] = np.argmax(tree_.value[node][i])   
            f.write("{}return {}".format(indent, pred_arr))
            f.write("\n")
            
    
    recurse(0, 1)
    f.close() 


# In[3]:


def file_len(fname):
    #i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def funcConvBranch(single_branch, dfT, rep, instances):
    
    f3 = open('DecSmt.smt2', 'a') 
    f3.write("(assert (=> (and ")
    
    noOfAttr = dfT.shape[1]
    for i in range(0, len(single_branch)):
        temp_Str = single_branch[i]
        if('if' in temp_Str):
            #temp_content[i] = content[i]
            for j in range (0, dfT.columns.values.shape[0]):
                if(dfT.columns.values[j] in temp_Str):
                    fe_name = str(dfT.columns.values[j])
                    fe_index = j
            data_type = str(dfT.dtypes[fe_index])
            if('<=' in temp_Str):
                sign = '<='
            elif('<=' in temp_Str):
                sign = '>'    
            elif('>' in temp_Str):
                sign = '>'
            elif('>=' in temp_Str):
                sign = '>='  
            #elif(('=') or ('==') in temp_Str):
            #    sign = '='
            
            split_arr = temp_Str.split(sign)
            split_arr = split_arr[1].strip()
            split_arr = split_arr.strip(':')
            
            if('int' in data_type):
                #digit = int(re.search(r'\d+', temp_Str).group(0))
                digit = int(split_arr)
            elif('float' in data_type):
                #digit = float(re.search(r'\d+', temp_Str).group(0))
                digit = float(split_arr)
            digit = str(digit)
            if(instances == 1):
                f3.write("(" + sign + " "+ fe_name +" " + digit +") ")  
            else:
                f3.write("(" + sign + " "+ fe_name +str(rep)+" " + digit +") ")
           
        elif('return' in temp_Str):
            class_array = np.array(re.findall(r'\d+', temp_Str))
            class_array =[int(k) for k in class_array]
            f3.write(") (and ")
            for k in range(0, len(class_array)):
                feature_name = dfT.columns.values[noOfAttr-len(class_array) +k]
                if(instances == 1):
                    f3.write("(= "+feature_name+" "+str(class_array[k])+")") 
                else:
                    f3.write("(= "+feature_name+str(rep)+" "+str(class_array[k])+")")          
            f3.write(")))")
            f3.write('\n')
            
    f3.close()
    


def funcGetBranch(sinBranch, dfT, rep, instances):
    flg = False
    for i in range (0, len(sinBranch)):
        tempSt = sinBranch[i]
        if('return' in tempSt):
            flg = True
            funcConvBranch(sinBranch, dfT, rep, instances)

def funcGenBranch(dfT, rep, instances):
 
    with open('TreeOutput.txt') as f1:
        file_content = f1.readlines()
    file_content = [x.strip() for x in file_content] 
    
    f1.close()
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    noOfAttr = dfT.shape[1]-int(paramDict['no_of_class']) 
    
    noOfLines = file_len('TreeOutput.txt')
    temp_file_cont = ["" for x in range(noOfLines)]
    
    i = 1
    k = 0
    while(i < noOfLines):
        
        j = k-1
        if(temp_file_cont[j] == '}'):
            funcGetBranch(temp_file_cont, dfT, rep, instances)
            while(True):
                if(temp_file_cont[j] == '{'):
                    temp_file_cont[j] = ''
                    temp_file_cont[j-1] = ''
                    j = j-1
                    break  
                elif(j>=0):    
                    temp_file_cont[j] = ''
                    j = j-1
            k = j    
        else:    
            temp_file_cont[k] = file_content[i]
            k = k+1
            i = i+1
           
    if('return' in file_content[1]):
        class_array = np.array(re.findall(r'\d+', file_content[1]))
        f = open('DecSmt.smt2', 'a')    
        class_array =[int(k) for k in class_array]
        f.write("(assert (and ")
        for k in range(0, len(class_array)):
            feature_name = dfT.columns.values[noOfAttr-len(class_array) +k]
            if(instances == 1):
                f.write("(= "+feature_name+" "+str(class_array[k])+")")
            else:
                f.write("(= "+feature_name+str(rep)+" "+str(class_array[k])+")")           
            
        f.write("))")
        f.write('\n')
    else:    
        funcGetBranch(temp_file_cont, dfT, rep, instances)


def funcConv(dfT, no_of_instances):

    temp_content = ['']
    rep = 0
    min_val = 0
    max_val = 0
    
    with open('TreeOutput.txt') as f1:
        content = f1.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
   
    noOfLines = file_len('TreeOutput.txt')
    #print(content.shape)
    
    f = open('DecSmt.smt2', 'w')
    for j in range(0, no_of_instances):
        for i in range (0, dfT.columns.values.shape[0]):
            tempStr = dfT.columns.values[i]
            fe_type = dfT.dtypes[i]
            fe_type = str(fe_type)
        
            min_val = dfT.iloc[:, i].min()
            max_val = dfT.iloc[:, i].max() 
        
            if('int' in fe_type):
                if(no_of_instances == 1):
                    f.write("(declare-fun " + tempStr+" () Int)")
                else:
                    f.write("(declare-fun " + tempStr+str(j)+ " () Int)")
                f.write('\n')
                #adding range
                #f2.write("(assert (and (>= "+tempStr+" "+str(min_val)+")"+" "+"(<= "+tempStr+"1 "+str(max_val)+")))")
                f.write('\n')
            elif('float' in fe_type):
                if(no_of_instances == 1):
                    f.write("(declare-fun " + tempStr+" () Real)")
                else:
                    f.write("(declare-fun " + tempStr+str(j)+ " () Real)")
                f.write('\n')
                #Adding range
                #f2.write("(assert (and (>= "+tempStr+"1 "+str(round(min_val, 2))+")"+" "+"(<= "+tempStr+"1 "+str(round(max_val, 2))+")))")
                f.write('\n') 
        f.write("; "+str(j)+"th element")
        f.write('\n')

    #Writing the functions for computing absolute integer & real value
    f.write('(define-fun absoluteInt ((x Int)) Int \n')
    f.write('  (ite (>= x 0) x (- x))) \n')

    f.write('(define-fun absoluteReal ((x Real)) Real \n')
    f.write('  (ite (>= x 0) x (- x))) \n')
        
    f.close()
    
    #Calling function to get the branch and convert it to z3 form,  creating alias
    for i in range(0, no_of_instances):  
        f = open('DecSmt.smt2', 'a')
        f.write('\n;-----------'+str(i)+'-----------number instance-------------- \n')
        f.close()
        funcGenBranch(dfT, i, no_of_instances)
   
  

def funcGenSMTFairness(df, no_of_instances):
    funcConv(df, no_of_instances)

def functree2LogicMain(tree, no_of_instances):
    df = pd.read_csv('OracleData.csv')
    tree_to_code(tree, df.columns)
    funcGenSMTFairness(df, no_of_instances)


    

