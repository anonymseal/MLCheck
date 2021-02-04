
import pandas as pd
import csv as cv
import sys
from sklearn import tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utils import util
import fileinput
import os
import re


def funcConvZ3OutToData(df):
    fe_flag = False

    with open('param_dict.csv') as csv_file:
        reader = cv.reader(csv_file)
        paramDict = dict(reader)
    no_of_params = int(paramDict['no_of_params'])
    testMatrix = np.zeros(((no_of_params), df.shape[1]))

    if(os.stat('FinalOutput.txt').st_size > 0):
        with open('FinalOutput.txt') as f1:
            file_content = f1.readlines()
    
        file_content = [x.strip() for x in file_content]
        noOfLines = util.file_len('FinalOutput.txt')

        with open('TestDataSMT.csv', 'w', newline='') as csvfile:
            fieldnames = df.columns.values  
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
            writer.writerows(testMatrix)

        dfAgain = pd.read_csv('TestDataSMT.csv')    
        nums = re.compile(r"[+-]?\d+(?:\.\d+)?")
        if('unknown' in file_content[0]):
            raise Exception('Encoding problem')
            sys.exit(1)
        if('model is not available' in file_content[1]):
            return False    
        else:
            i = 1
            while(i < noOfLines):
                minus_flag = False
                fe_flag = False         
                if("(model" == file_content[i]):
                    i = i+1
                elif(")" == file_content[i]):
                    i = i+1
                else:
                    for j in range (0, df.columns.values.shape[0]):
                        for param_no in range(0, no_of_params):    
                            if(paramDict['multi_label'] == 'True' and no_of_params == 1 and paramDict['white_box_model'] == 'Decision tree'):
                                fe_add = ' '
                            else:
                                fe_add = str(param_no)
                            if(df.columns.values[j]+fe_add in file_content[i]):
                                feature_name = df.columns.values[j]
                                fe_flag = True
                                if('Int' in file_content[i]):
                                    i = i+1
                                    digit = int(re.search(r'\d+', file_content[i]).group(0))
                                    if('-' in file_content[i]):
                                        digit = 0-digit
                                elif('Real' in file_content[i]):
                                    i = i+1
                                    if("(/" in file_content[i]):
                                        if('-' in file_content[i]):
                                            minus_flag = True
                                        multi_digits = re.findall('\d*?\.\d+', file_content[i])
                                        if(len(multi_digits) == 1):
                                            i = i+1
                                            multi_digits.append(float(re.search(r'\d+', file_content[i]).group(0)))
                                        digit = float(multi_digits[0])/float(multi_digits[1])
                                        if(minus_flag == True):
                                            digit = 0-digit
                                    else:   
                                        #digit = round(float(re.search(r'\d+', file_content[i]).group(0)), 2)
                                        digit = float(re.search(r'\d+', file_content[i]).group(0))
                                        if('-' in file_content[i]):
                                            digit = 0-digit
                                dfAgain.loc[param_no, feature_name] = digit
                                i=i+1
                    if(fe_flag == False):
                        i = i+2                         
            dfAgain.to_csv('TestDataSMT.csv', index= False, header=True)
            return True
    
    else:
        raise Exception("There is no solver installed in your system")


funcConvZ3OutToData
            
            

