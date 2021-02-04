
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def mon_indxArr():
    
    noOfLines = file_len('testFeature.txt')

    with open('testFeature.txt') as f1:
        file_content = f1.readlines()
    file_content = [x.strip() for x in file_content] 
    f1.close()    

    with open('DataFile.txt') as f2:
        data_file_content = f2.readline()
    data_file_content = str(data_file_content)
    f2.close()
    df=pd.read_csv(data_file_content)
    
    index_arr = np.zeros((noOfLines, ))
    
    for i in range (0, len(file_content)):
        for j in range(0, df.shape[1]-1):
            if(file_content[i] in df.columns.values[j]):
                index_arr[i] = j
    
    return index_arr

def predict(model, x):
    arr = np.zeros((1, len(x)))
    for i in range (0, len(x)):
        arr[0][i] = x[i]
        
    #print(model.predict(arr))    
    return model.predict(arr)

