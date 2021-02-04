from joblib import dump, load
import numpy as np
import csv as cv
import pandas as pd
import os

def String2List(string):
    return list(string.split(" "))

def file_len(fname):
    if(os.stat(fname).st_size == 0):
        return 'empty'
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def convDataInst(X, df, j, no_of_class):
    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    if(paramDict['multi_label'] == 'False'):
        no_of_class = 1
    data_inst= np.zeros((1, df.shape[1]-no_of_class))
    if(j > X.shape[0]):
        raise Exception('Z3 has produced counter example with all 0 values of the features: Run the script Again')
        sys.exit(1)
    for i in range(df.shape[1]-no_of_class):
        data_inst[0][i] = X[j][i]
    return data_inst

def funcAdd2Oracle(data):    
    with open('TestingData.csv', 'a', newline='') as csvfile:  
        writer = cv.writer(csvfile)
        writer.writerows(data)


def funcCreateOracle(no_of_class, multi_label, model):
    df = pd.read_csv('TestingData.csv')
    data = df.values
    if multi_label == 'False':
        X = data[:, :-1]
        predict_class = model.predict(X)
        for i in range(0, X.shape[0]):
            df.loc[i, 'Class'] = predict_class[i]
    else:
        X = data[:, :-no_of_class]
        predict_class = model.predict(X)
        index = df.shape[1]-no_of_class
        for i in range(0, no_of_class):
            className = str(df.columns.values[index+i])
            for j in range(0, X.shape[0]):
                df.loc[j, className] = predict_class[j][i]
    df.to_csv('OracleData.csv', index=False, header=True)


def storeMapping(file_name, dictionary):
    try:
        with open(file_name, 'w') as csv_file:  
            writer = cv.writer(csv_file)
            for key, value in dictionary.items():
                writer.writerow([key, value])
    except IOError:
        print("I/O error")

def addContent(file_name, f_content): 
    f1 = open(file_name, 'a')
    for x in f_content:
        f1.write('\n')
        f1.write(x)
    f1.close()

def addSatOpt(file_name):
    f = open(file_name, 'a')
    f.write('\n')
    f.write("(check-sat) \n")
    f.write("(get-model) \n")

def storeAssumeAssert(file_name):
    with open('dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        assumeDict = dict(reader)
    if(assumeDict['no_assumption'] == 'True'):
        pass
    else:
        with open('assumeStmnt.txt') as f2:
            f2_content = f2.readlines()
        f2_content = [x.strip() for x in f2_content]
        addContent(file_name, f2_content)
    with open('assertStmnt.txt') as f3:
        f3_content = f3.readlines()
    f3_content = [x.strip() for x in f3_content]
    addContent(file_name, f3_content)



